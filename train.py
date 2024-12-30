import glob
import os

import matplotlib
import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
from sklearn.metrics import accuracy_score, recall_score, f1_score

matplotlib.use('Agg')  # 设置后端为 Agg
import matplotlib.pyplot as plt
import cv2

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import queue
import datetime
import numpy as np

# 1. 数据集类
class UCSDDataset(paddle.io.Dataset):
    def __init__(self, root_path, is_train=True, transform=None, clip_length=16):
        super(UCSDDataset, self).__init__()
        self.root_path = root_path
        self.is_train = is_train
        self.transform = transform
        self.clip_length = clip_length

        # 确定数据目录
        self.data_path = os.path.join(root_path, 'Train' if is_train else 'Test')

        # 预加载所有数据到内存
        self.frames_data = []  # 存储预处理后的帧数据
        self.labels = []

        print(f"开始加载{'训练' if is_train else '测试'}数据...")

        # 遍历所有序列目录
        sequence_dirs = sorted([d for d in os.listdir(self.data_path)
                                if os.path.isdir(os.path.join(self.data_path, d)) and 'gt' not in d])

        for seq_dir in sequence_dirs:
            seq_path = os.path.join(self.data_path, seq_dir)
            # 获取该序列中的所有帧
            frames = sorted(glob.glob(os.path.join(seq_path, '*.tif')))

            # 将帧分组为固定长度的片段
            for i in range(0, len(frames) - self.clip_length + 1, self.clip_length):
                clip_frames = frames[i:i + self.clip_length]
                if len(clip_frames) == self.clip_length:
                    # 预处理并存储帧数据
                    processed_frames = []
                    for frame_path in clip_frames:
                        # 读取图像并转换为RGB
                        frame = Image.open(frame_path).convert('RGB')
                        if self.transform:
                            frame = self.transform(frame)
                        processed_frames.append(frame)

                    # 将处理后的帧数据存储为张量
                    clip_tensor = paddle.stack(processed_frames)
                    self.frames_data.append(clip_tensor)

                    # 确定标签
                    if is_train:
                        self.labels.append(0)  # 训练集全部标记为正常
                    else:
                        # 测试集根据是否有对应的gt文件夹判断
                        gt_dir = seq_dir + '_gt'
                        if os.path.exists(os.path.join(self.data_path, gt_dir)):
                            self.labels.append(1)  # 异常
                        else:
                            self.labels.append(0)  # 正常

        print(f"数据加载完成！共加载 {len(self.frames_data)} 个视频片段")

    def __getitem__(self, idx):
        """直接返回预加载的数据"""
        return self.frames_data[idx], paddle.to_tensor([self.labels[idx]], dtype='int64')

    def __len__(self):
        return len(self.frames_data)


# 2. 数据预处理
class VideoPreprocessor:
    def __init__(self, clip_length=16, frame_size=(128, 128)):
        self.clip_length = clip_length
        self.frame_size = frame_size

    def split_video(self, video_path):
        """将长视频分割成固定长度的片段"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()

        # 将frames分割成固定长度的片段
        clips = [frames[i:i+self.clip_length] for i in range(0, len(frames), self.clip_length)]
        return clips

    def extract_keyframes(self, frames, method='uniform'):
        """从视频片段中提取关键帧"""
        if method == 'uniform':
            # 均匀采样
            step = len(frames) // self.clip_length
            return frames[::step][:self.clip_length]

    def augment_data(self, frame):
        """数据增强"""
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(self.frame_size, scale=(0.8, 1.0))
        ]
        for aug in augmentations:
            frame = aug(frame)
        return frame

# 3. 特征提取器
class FeatureExtractor(nn.Layer):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        # 使用预训练的ResNet作为特征提取器
        self.backbone = paddle.vision.models.resnet50(pretrained=pretrained)
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        return self.backbone(x)
# 4. 异常检测模型
class AnomalyDetector(nn.Layer):
    def __init__(self, input_size=2048, hidden_size=512):
        super(AnomalyDetector, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.temporal_model = nn.LSTM(input_size, hidden_size, num_layers=2, direction='bidirect')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.reshape([-1, channels, height, width])  # 确保通道数正确

        # 提取特征
        features = self.feature_extractor(x)
        features = features.reshape([batch_size, time_steps, -1])

        # 时序建模
        temporal_features, _ = self.temporal_model(features)

        # 分类
        output = self.classifier(temporal_features[:, -1])
        return output

# 5. 训练和评估系统
class AnomalyDetectionSystem:
    def __init__(self, model, device='gpu'):
        self.model = model
        self.device = device
        self.metrics = {}

    def train(self, train_loader, val_loader, epochs=50):
        optimizer = paddle.optimizer.Adam(parameters=self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        metrics_history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_recall': [],
            'val_f1': []
        }

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.clear_grad()
                output = self.model(data)
                loss = criterion(output, target.squeeze())  # 修改这里，确保target维度正确
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss/len(train_loader)
            metrics_history['train_loss'].append(avg_train_loss)

            # 验证阶段
            val_metrics = self.evaluate(val_loader)
            metrics_history['val_accuracy'].append(val_metrics['accuracy'])
            metrics_history['val_recall'].append(val_metrics['recall'])
            metrics_history['val_f1'].append(val_metrics['f1'])

            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, '
                  f'Val Accuracy = {val_metrics["accuracy"]:.4f}')

        return metrics_history

    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        targets = []

        with paddle.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                pred = output.argmax(axis=1)
                predictions.extend(pred.numpy())
                targets.extend(target.squeeze().numpy())

        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'recall': recall_score(targets, predictions, average='macro'),
            'f1': f1_score(targets, predictions, average='macro')
        }
        return metrics

# 6. 可视化模块
class ResultVisualizer:
    def __init__(self):
        plt.switch_backend('Agg')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

    def plot_detection_results(self, video_frames, predictions, timestamps):
        """可视化检测结果"""
        for i, (frame, pred, ts) in enumerate(zip(video_frames, predictions, timestamps)):
            if pred == 1:  # 异常行为
                self.ax.clear()
                self.ax.imshow(frame)
                self.ax.set_title(f'Anomaly Detected at {ts}')
                plt.savefig(f'anomaly_frame_{i}.png')

    def plot_metrics(self, metrics_history):
        """绘制训练过程中的指标变化"""
        plt.figure(figsize=(10, 5))
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
        plt.legend()
        plt.title('Training Metrics')
        plt.savefig('training_metrics.png')
        plt.close()


class AnomalyDetectionGUI:
    def __init__(self, model):
        self.model = model
        self.setup_gui()
        self.setup_video_processing()

    def setup_gui(self):
        """初始化GUI界面"""
        self.window = tk.Tk()
        self.window.title("视频异常行为检测系统")
        self.window.geometry("1200x800")

        # 创建主框架
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 视频源选择
        ttk.Label(self.control_frame, text="选择视频源:").pack(pady=5)
        self.source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(self.control_frame, text="摄像头", variable=self.source_var,
                        value="camera").pack()
        ttk.Radiobutton(self.control_frame, text="视频文件", variable=self.source_var,
                        value="file").pack()

        # 控制按钮
        self.start_button = ttk.Button(self.control_frame, text="开始检测",
                                       command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self.control_frame, text="停止检测",
                                      command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.file_button = ttk.Button(self.control_frame, text="选择视频文件",
                                      command=self.select_file)
        self.file_button.pack(pady=5)

        # 检测阈值调节
        ttk.Label(self.control_frame, text="检测阈值:").pack(pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(self.control_frame, from_=0, to=1,
                                         variable=self.threshold_var, orient=tk.HORIZONTAL)
        self.threshold_scale.pack(pady=5, fill=tk.X, padx=5)

        # 右侧显示区域
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # 视频显示
        self.video_label = ttk.Label(self.display_frame)
        self.video_label.pack(pady=5)

        # 检测结果显示
        self.result_frame = ttk.LabelFrame(self.display_frame, text="检测结果")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建结果文本框和滚动条
        self.result_text = tk.Text(self.result_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL,
                                  command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(self.window, textvariable=self.status_var,
                                    relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_video_processing(self):
        """初始化视频处理相关变量"""
        self.is_running = False
        self.video_source = None
        self.frame_queue = queue.Queue(maxsize=128)
        self.result_queue = queue.Queue()
        self.video_thread = None
        self.detection_thread = None

    def select_file(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"已选择文件: {file_path}")

    def start_detection(self):
        """开始检测"""
        if self.source_var.get() == "camera":
            self.video_source = cv2.VideoCapture(0)
        else:
            if hasattr(self, 'video_path'):
                self.video_source = cv2.VideoCapture(self.video_path)
            else:
                self.status_var.set("请先选择视频文件！")
                return

        if not self.video_source.isOpened():
            self.status_var.set("无法打开视频源！")
            return

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("正在检测...")

        # 启动视频处理线程
        self.video_thread = threading.Thread(target=self.process_video)
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.video_thread.start()
        self.detection_thread.start()

        # 开始更新GUI
        self.update_gui()

    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        if self.video_source:
            self.video_source.release()

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("检测已停止")

    def process_video(self):
        """处理视频帧"""
        frames_buffer = []
        while self.is_running:
            ret, frame = self.video_source.read()
            if not ret:
                break

            frames_buffer.append(frame)
            if len(frames_buffer) == 16:  # 积累足够的帧
                # 将帧序列放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put((frames_buffer.copy(), frame))
                frames_buffer.pop(0)

    def run_detection(self):
        """运行检测"""
        while self.is_running:
            if not self.frame_queue.empty():
                frames_buffer, current_frame = self.frame_queue.get()

                # 在这里添加实际的检测逻辑
                # 使用self.model进行预测
                # prediction = self.model.predict(frames_buffer)

                # 模拟检测结果
                prediction = np.random.choice([0, 1], p=[0.8, 0.2])

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.result_queue.put({
                    'frame': current_frame,
                    'prediction': prediction,
                    'timestamp': timestamp
                })

    def update_gui(self):
        """更新GUI显示"""
        if not self.result_queue.empty():
            result = self.result_queue.get()
            frame = result['frame']
            prediction = result['prediction']
            timestamp = result['timestamp']

            # 更新视频显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            if prediction == 1:  # 检测到异常
                cv2.putText(frame, "Anomaly Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.config(image=photo)
            self.video_label.image = photo

            # 更新检测结果
            if prediction == 1:
                self.result_text.insert(tk.END,
                                        f"[{timestamp}] 检测到异常行为！\n")
                self.result_text.see(tk.END)

        if self.is_running:
            self.window.after(10, self.update_gui)


def main():
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("正在加载数据集...")
    train_dataset = UCSDDataset(
        root_path='./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1',
        is_train=True,
        transform=transform,
        clip_length=16
    )
    train_dataset.frames_data = train_dataset.frames_data[:2]
    train_dataset.labels = train_dataset.labels[:2]
    # 增加批处理大小，减少迭代次数
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=8,  # 增加批处理大小
        shuffle=True,
        num_workers=0
    )

    print("数据加载完成，开始训练...")

    # 初始化模型和训练系统
    model = AnomalyDetector()
    system = AnomalyDetectionSystem(model)
    visualizer = ResultVisualizer()

    # 训练模型
    metrics_history = system.train(train_loader, train_loader, epochs=2)

    # 可视化训练结果
    visualizer.plot_metrics(metrics_history)

    # 保存模型
    paddle.save(model.state_dict(), 'anomaly_detector_final.pdparams')
    gui = AnomalyDetectionGUI(model)  # 传入训练好的模型
    gui.window.mainloop()



if __name__ == '__main__':
    main()
