#!/usr/bin/env python3

"""
EchoSense - 音频处理综合应用程序

功能：
1. 音频文件拖拽导入（带进度条）
2. 音频降噪处理
3. 性别识别
4. 人声分析（声学特征优先）
5. 说话人识别

使用PyQt5构建单窗口界面，集成所有功能模块
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import soundfile as sf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QPushButton, QLabel, QFileDialog, QProgressBar,
    QSlider, QTextEdit, QComboBox, QGridLayout, QGroupBox,
    QListWidget, QSplitter, QSizePolicy, QCheckBox, QScrollArea,
    QDialog, QTextBrowser
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeNoise.denoiser import Denoiser
from Gender.GPU_GenderRecognition import GPUGenderAnalyzer
from VoiceAnalysis.voice_analyzer import VoiceAnalyzer
from VoiceAnalysis.logger import setup_logger, log_gui_operation, log_error
from EmotionAnalysis.emotion_analyzer import EmotionAnalyzer
from EmotionAnalysis.emotion_visualizer import EmotionVisualizer

# 添加ModelTrain到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ModelTrain'))

# 添加ECAPA-TDNN到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ECAPA-TDNN'))

app_logger = setup_logger('AudioApp')


class WorkerThread(QThread):
    """工作线程，用于执行耗时操作"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainingWorkerThread(QThread):
    """训练工作线程，支持进度回调"""
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model_type="random_forest", max_samples=None, max_speakers=None):
        super().__init__()
        self.model_type = model_type
        self.max_samples = max_samples
        self.max_speakers = max_speakers
        self._is_running = True
    
    def run(self):
        try:
            def progress_callback(progress_dict):
                if self._is_running:
                    self.progress.emit(progress_dict)
            
            result = train_from_scratch(
                model_type=self.model_type,
                max_samples=self.max_samples,
                max_speakers=self.max_speakers,
                progress_callback=progress_callback
            )
            
            if self._is_running:
                self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class VoiceAnalysisWorkerThread(QThread):
    """人声分析工作线程，支持进度回调"""
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, analyzer, audio_file, analysis_mode):
        super().__init__()
        self.analyzer = analyzer
        self.audio_file = audio_file
        self.analysis_mode = analysis_mode
        self._is_running = True
    
    def run(self):
        try:
            def progress_callback(progress_dict):
                if self._is_running:
                    self.progress.emit(progress_dict)
            
            result = self.analyzer.analyze_audio(
                self.audio_file,
                generate_report=True,
                progress_callback=progress_callback
            )
            
            if self._is_running:
                self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class DenoiseWorkerThread(QThread):
    """降噪工作线程，支持进度回调"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, denoiser, audio_file):
        super().__init__()
        self.denoiser = denoiser
        self.audio_file = audio_file
        self._is_running = True
    
    def run(self):
        try:
            def progress_callback(progress, message):
                if self._is_running:
                    self.progress.emit(progress, message)
            
            audio, sr = librosa.load(self.audio_file, sr=16000)
            
            denoised = self.denoiser.denoise(audio, sr, progress_callback)
            
            if self._is_running:
                self.finished.emit((audio, denoised, sr))
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class EcapaWorker(QThread):
    """ECAPA-TDNN 说话人分析工作线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            from ecapa_speaker import analyze_speakers
            result = analyze_speakers(self.audio_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MplCanvas(FigureCanvas):
    """Matplotlib画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        
        self.tooltip_info = []
        self.tooltip = None
        
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('figure_leave_event', self.on_mouse_leave)
    
    def on_mouse_move(self, event):
        """鼠标移动事件处理"""
        if event.inaxes != self.axes:
            if self.tooltip:
                self.tooltip.remove()
                self.tooltip = None
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if not self.tooltip_info:
            if self.tooltip:
                self.tooltip.remove()
                self.tooltip = None
            return
        
        closest_idx = min(range(len(self.tooltip_info)), 
                         key=lambda i: abs(self.tooltip_info[i]['time'] - x))
        
        time_diff = abs(self.tooltip_info[closest_idx]['time'] - x)
        if time_diff > 0.5:
            if self.tooltip:
                self.tooltip.remove()
                self.tooltip = None
            return
        
        info = self.tooltip_info[closest_idx]
        tooltip_text = f"时间: {info['time']:.1f}s\n"
        tooltip_text += f"说话人: {info['speaker']}\n"
        if 'gender' in info:
            tooltip_text += f"性别: {info['gender']}\n"
        if 'energy' in info:
            tooltip_text += f"能量: {info['energy']:.4f}\n"
        if 'zcr' in info:
            tooltip_text += f"过零率: {info['zcr']:.4f}\n"
        if 'similarity' in info:
            tooltip_text += f"相似度: {info['similarity']:.4f}\n"
        
        if self.tooltip:
            self.tooltip.set_text(tooltip_text)
            self.tooltip.set_x(x)
            self.tooltip.set_y(y)
        else:
            self.tooltip = self.axes.annotate(
                tooltip_text,
                xy=(x, y),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->')
            )
        
        self.draw()
    
    def on_mouse_leave(self, event):
        """鼠标离开事件处理"""
        if self.tooltip:
            self.tooltip.remove()
            self.tooltip = None
            self.draw()
    
    def set_tooltip_info(self, info):
        """设置悬停提示信息"""
        self.tooltip_info = info


class AudioTab(QWidget):
    """音频导入标签页"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        
        self.drop_area = QGroupBox("拖拽音频文件到此处")
        drop_layout = QVBoxLayout()
        self.drop_label = QLabel("或点击按钮选择文件")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("border: 2px dashed #ccc; padding: 40px; border-radius: 10px;")
        drop_layout.addWidget(self.drop_label)
        self.drop_area.setLayout(drop_layout)
        
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("选择文件")
        self.select_button.clicked.connect(self.select_file)
        self.delete_button = QPushButton("删除选中文件")
        self.delete_button.clicked.connect(self.delete_file)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.delete_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setMinimumHeight(100)
        
        self.audio_list = QListWidget()
        self.audio_list.setMinimumHeight(150)
        
        self.layout.addWidget(self.drop_area)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(QLabel("加载进度:"))
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("已加载的音频文件:"))
        self.layout.addWidget(self.audio_list)
        self.layout.addWidget(QLabel("文件信息:"))
        self.layout.addWidget(self.file_info)
        
        self.setAcceptDrops(True)
        self.drop_area.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                self.load_audio_file(file_path)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", ".", "音频文件 (*.wav *.mp3 *.flac *.ogg)"
        )
        if file_path:
            self.load_audio_file(file_path)
    
    def load_audio_file(self, file_path):
        try:
            self.progress_bar.setValue(10)
            self.progress_bar.setFormat("正在加载...")
            QApplication.processEvents()
            
            y, sr = librosa.load(file_path, sr=None)
            
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            duration = len(y) / sr
            
            info = f"文件路径: {file_path}\n"
            info += f"文件名: {os.path.basename(file_path)}\n"
            info += f"采样率: {sr} Hz\n"
            info += f"时长: {duration:.2f} 秒\n"
            info += f"通道数: 1 (单声道)\n"
            self.file_info.setText(info)
            
            self.progress_bar.setValue(80)
            QApplication.processEvents()
            
            self.audio_list.addItem(os.path.basename(file_path))
            
            self.parent.current_audio = file_path
            
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("加载完成!")
            self.parent.statusBar().showMessage(f"已加载音频文件: {os.path.basename(file_path)}")
        except Exception as e:
            self.progress_bar.setFormat("加载失败!")
            self.parent.statusBar().showMessage(f"加载失败: {str(e)}")
    
    def delete_file(self):
        """删除选中的音频文件"""
        selected_items = self.audio_list.selectedItems()
        if not selected_items:
            self.parent.statusBar().showMessage("请先选择要删除的文件")
            return
        
        for item in selected_items:
            self.audio_list.takeItem(self.audio_list.row(item))
        
        if self.audio_list.count() == 0:
            self.file_info.clear()
            self.parent.current_audio = None
        
        self.parent.statusBar().showMessage(f"已删除 {len(selected_items)} 个文件")


class DenoiseTab(QWidget):
    """音频降噪标签页"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("降噪算法:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["传统谱减法", "高级降噪（推荐）", "深度学习降噪"])
        self.algorithm_combo.setCurrentText("高级降噪（推荐）")
        param_layout.addWidget(self.algorithm_combo, 0, 1, 1, 2)
        
        param_layout.addWidget(QLabel("噪声减少程度:"), 1, 0)
        self.noise_reduction_slider = QSlider(Qt.Horizontal)
        self.noise_reduction_slider.setRange(10, 90)
        self.noise_reduction_slider.setValue(30)
        self.noise_reduction_label = QLabel("30%")
        self.noise_reduction_slider.valueChanged.connect(
            lambda value: self.noise_reduction_label.setText(f"{value}%")
        )
        param_layout.addWidget(self.noise_reduction_slider, 1, 1)
        param_layout.addWidget(self.noise_reduction_label, 1, 2)
        
        param_layout.addWidget(QLabel("批处理大小:"), 2, 0)
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(["16", "32", "64", "128"])
        self.batch_size_combo.setCurrentText("32")
        param_layout.addWidget(self.batch_size_combo, 2, 1, 1, 2)
        
        button_layout = QHBoxLayout()
        self.denoise_button = QPushButton("开始降噪")
        self.denoise_button.clicked.connect(self.start_denoise)
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.denoise_button)
        button_layout.addWidget(self.save_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(100)
        
        self.canvas = MplCanvas(self, width=8, height=4)
        
        self.layout.addLayout(param_layout)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("处理状态:"))
        self.layout.addWidget(self.status_text)
        self.layout.addWidget(QLabel("音频对比:"))
        self.layout.addWidget(self.canvas)
        
        self.denoiser = None
        self.denoised_audio = None
        self.sr = None
    
    def start_denoise(self):
        if not self.parent.current_audio:
            self.parent.statusBar().showMessage("请先加载音频文件")
            return
        
        algorithm = self.algorithm_combo.currentText()
        noise_reduction = self.noise_reduction_slider.value() / 100
        batch_size = int(self.batch_size_combo.currentText())
        
        algorithm_map = {
            "传统谱减法": "basic",
            "高级降噪（推荐）": "advanced",
            "深度学习降噪": "deep"
        }
        algo = algorithm_map[algorithm]
        
        self.denoiser = Denoiser(algorithm=algo)
        
        self.denoise_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()
        self.status_text.append(f"开始降噪处理...")
        self.status_text.append(f"降噪算法: {algorithm}")
        self.status_text.append(f"噪声减少程度: {noise_reduction:.2f}")
        self.status_text.append(f"批处理大小: {batch_size}")
        self.status_text.append("")
        
        self.worker = DenoiseWorkerThread(self.denoiser, self.parent.current_audio)
        self.worker.progress.connect(self.update_denoise_progress)
        self.worker.finished.connect(self.denoise_finished)
        self.worker.error.connect(self.denoise_error)
        self.worker.start()
    
    def update_denoise_progress(self, progress, message):
        """更新降噪进度"""
        self.progress_bar.setValue(progress)
        self.status_text.append(f"[{progress:3d}%] {message}")
        # 滚动到最新行
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)
    
    def denoise_finished(self, result):
        original, denoised, sr = result
        
        self.status_text.append("降噪完成！")
        self.save_button.setEnabled(True)
        self.denoise_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.parent.statusBar().showMessage("降噪处理完成")
        
        self.canvas.axes.clear()
        
        time_orig = np.arange(len(original)) / sr
        self.canvas.axes.plot(time_orig[:10000], original[:10000], label="原始音频")
        
        time_denoised = np.arange(len(denoised)) / sr
        self.canvas.axes.plot(time_denoised[:10000], denoised[:10000], label="降噪后音频")
        
        self.canvas.axes.set_title("音频对比")
        self.canvas.axes.set_xlabel("时间 (秒)")
        self.canvas.axes.set_ylabel("振幅")
        self.canvas.axes.legend()
        self.canvas.draw()
    
    def denoise_error(self, error_msg):
        self.status_text.append(f"错误: {error_msg}")
        self.denoise_button.setEnabled(True)
        self.parent.statusBar().showMessage(f"处理失败: {error_msg}")
    
    def save_result(self):
        if self.denoised_audio is None or len(self.denoised_audio) == 0:
            return
        
        output_file, _ = QFileDialog.getSaveFileName(
            self, "保存降噪结果", ".", "音频文件 (*.wav *.mp3)"
        )
        if output_file:
            try:
                sf.write(output_file, self.denoised_audio, self.sr)
                self.parent.statusBar().showMessage(f"结果已保存: {output_file}")
            except Exception as e:
                self.parent.statusBar().showMessage(f"保存失败: {str(e)}")


class GenderTab(QWidget):
    """性别识别标签页"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(150)
        
        self.canvas = MplCanvas(self, width=8, height=5)
        
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("分析结果:"))
        self.layout.addWidget(self.result_text)
        self.layout.addWidget(QLabel("性别评分时间序列:"))
        self.layout.addWidget(self.canvas)
        
        self.analyzer = GPUGenderAnalyzer()
    
    def start_analysis(self):
        if not self.parent.current_audio:
            self.parent.statusBar().showMessage("请先加载音频文件")
            return
        
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.result_text.append("开始性别识别分析...")
        
        def analyze_task():
            gender_scores, processing_times, total_time = self.analyzer.analyze_audio_gpu(self.parent.current_audio)
            
            avg_score = np.mean(gender_scores)
            if avg_score >= 0.6:
                gender = "女性"
                confidence = avg_score * 100
            elif avg_score <= 0.4:
                gender = "男性"
                confidence = (1 - avg_score) * 100
            else:
                gender = "中性/不确定"
                confidence = max(avg_score, 1 - avg_score) * 100
            
            female_segments = sum(1 for score in gender_scores if score >= 0.6)
            male_segments = sum(1 for score in gender_scores if score <= 0.4)
            neutral_segments = len(gender_scores) - female_segments - male_segments
            
            return gender_scores, processing_times, total_time, gender, confidence, female_segments, male_segments, neutral_segments
        
        self.worker = WorkerThread(analyze_task)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
    
    def analysis_finished(self, result):
        gender_scores, processing_times, total_time, gender, confidence, female_segments, male_segments, neutral_segments = result
        
        self.result_text.append("性别分布:")
        self.result_text.append(f"女性特征段: {female_segments} ({female_segments/len(gender_scores)*100:.1f}%)")
        self.result_text.append(f"男性特征段: {male_segments} ({male_segments/len(gender_scores)*100:.1f}%)")
        self.result_text.append(f"中性特征段: {neutral_segments} ({neutral_segments/len(gender_scores)*100:.1f}%)")
        
        self.canvas.axes.clear()
        
        segments = range(len(gender_scores))
        
        male_scores = [score if score <= 0.4 else None for score in gender_scores]
        self.canvas.axes.plot(segments, male_scores, 'b-', linewidth=2, label='男声特征')
        
        female_scores = [score if score >= 0.6 else None for score in gender_scores]
        self.canvas.axes.plot(segments, female_scores, 'r-', linewidth=2, label='女声特征')
        
        neutral_scores = [score if 0.4 < score < 0.6 else None for score in gender_scores]
        self.canvas.axes.plot(segments, neutral_scores, 'g-', linewidth=2, label='中性特征')
        
        self.canvas.axes.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='女性阈值')
        self.canvas.axes.axhline(y=0.4, color='b', linestyle='--', alpha=0.5, label='男性阈值')
        
        self.canvas.axes.set_title('性别评分时间序列')
        self.canvas.axes.set_xlabel('音频段序号')
        self.canvas.axes.set_ylabel('性别评分')
        self.canvas.axes.legend()
        self.canvas.draw()
        
        self.analyze_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.parent.statusBar().showMessage("性别识别分析完成")
    
    def analysis_error(self, error_msg):
        self.result_text.append(f"错误: {error_msg}")
        self.analyze_button.setEnabled(True)
        self.parent.statusBar().showMessage(f"分析失败: {error_msg}")


class VoiceAnalysisTab(QWidget):
    """人声分析标签页"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("分析模式:"), 0, 0)
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(["完整分析", "仅声学特征", "仅情绪分析"])
        self.analysis_mode_combo.setCurrentText("完整分析")
        param_layout.addWidget(self.analysis_mode_combo, 0, 1, 1, 2)
        
        param_layout.addWidget(QLabel("并行处理:"), 1, 0)
        self.parallel_checkbox = QCheckBox()
        self.parallel_checkbox.setChecked(True)
        param_layout.addWidget(self.parallel_checkbox, 1, 1, 1, 2)
        
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.export_result)
        self.export_button.setEnabled(False)
        self.help_button = QPushButton("特征说明")
        self.help_button.clicked.connect(self.show_feature_help)
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.help_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(150)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #0c0c0c;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #00ff00;
                padding: 5px;
            }
        """)
        self.clear_console_button = QPushButton("清空控制台")
        self.clear_console_button.clicked.connect(self.clear_console)
        console_button_layout = QHBoxLayout()
        console_button_layout.addStretch()
        console_button_layout.addWidget(self.clear_console_button)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        
        self.layout.addLayout(param_layout)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(QLabel("分析控制台:"))
        self.layout.addWidget(self.console_output)
        self.layout.addLayout(console_button_layout)
        self.layout.addWidget(QLabel("分析结果:"))
        self.layout.addWidget(self.result_text)
        self.layout.addWidget(QLabel("可视化分析:"))
        self.layout.addWidget(self.scroll_area)
        
        self.analyzer = VoiceAnalyzer(
            use_parallel=self.parallel_checkbox.isChecked(),
            log_dir=log_dir
        )
        self.current_result = None
    
    def start_analysis(self):
        if not self.parent.current_audio:
            self.parent.statusBar().showMessage("请先加载音频文件")
            return
        
        use_parallel = self.parallel_checkbox.isChecked()
        analysis_mode = self.analysis_mode_combo.currentText()
        
        log_gui_operation(app_logger, "开始人声分析", 
                          f"文件: {os.path.basename(self.parent.current_audio)}, "
                          f"模式: {analysis_mode}, 并行: {'是' if use_parallel else '否'}")
        
        self.analyzer = VoiceAnalyzer(
            use_parallel=use_parallel,
            log_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        )
        
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.console_output.clear()
        self.result_text.append("开始人声分析...")
        self.result_text.append(f"分析模式: {analysis_mode}")
        self.result_text.append(f"并行处理: {'启用' if use_parallel else '禁用'}")
        self.result_text.append("")
        
        self.worker = VoiceAnalysisWorkerThread(self.analyzer, self.parent.current_audio, analysis_mode)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
    
    def analysis_finished(self, result):
        log_gui_operation(app_logger, "人声分析完成",
                          f"结果: {'成功' if result else '失败'}")
        
        if result:
            self.current_result = result
            
            self.result_text.append("=" * 60)
            self.result_text.append("人声分析完成！")
            self.result_text.append("=" * 60)
            self.result_text.append("")
            
            self.result_text.append("【音频信息】")
            self.result_text.append(f"  文件: {os.path.basename(result['audio_file'])}")
            self.result_text.append(f"  时长: {result['duration']:.2f}秒")
            self.result_text.append(f"  采样率: {result['sample_rate']}Hz")
            self.result_text.append(f"  分析段数: {result['num_segments']}")
            self.result_text.append("")
            
            features = result['overall_features']
            
            self.result_text.append("【声学特征】")
            f0_features = features.get('f0', {})
            if f0_features:
                self.result_text.append(f"  平均基频: {f0_features.get('mean_f0_mean', 0):.1f} Hz")
                self.result_text.append(f"  基频范围: {f0_features.get('range_f0_mean', 0):.1f} Hz")
                self.result_text.append(f"  基频标准差: {f0_features.get('std_f0_mean', 0):.1f} Hz")
            
            spectral_features = features.get('spectral', {})
            if spectral_features:
                self.result_text.append(f"  频谱质心: {spectral_features.get('spectral_centroid_mean_mean', 0):.1f} Hz")
                self.result_text.append(f"  频谱带宽: {spectral_features.get('spectral_bandwidth_mean_mean', 0):.1f} Hz")
            
            energy_features = features.get('energy', {})
            if energy_features:
                self.result_text.append(f"  RMS能量: {energy_features.get('rms_mean_mean', 0):.4f}")
                self.result_text.append(f"  能量动态范围: {energy_features.get('dynamic_range_mean', 0):.4f}")
            
            quality_features = features.get('quality', {})
            if quality_features:
                self.result_text.append(f"  HNR: {quality_features.get('hnr_mean', 0):.1f} dB")
                self.result_text.append(f"  Jitter: {quality_features.get('jitter_mean', 0):.2f}%")
                self.result_text.append(f"  Shimmer: {quality_features.get('shimmer_mean', 0):.2f}%")
            self.result_text.append("")
            
            self.plot_results(result)
            
            self.export_button.setEnabled(True)
            self.progress_bar.setValue(100)
            self.parent.statusBar().showMessage("人声分析完成")
        else:
            self.result_text.append("分析失败！")
        
        self.analyze_button.setEnabled(True)
    
    def plot_results(self, result):
        """绘制分析结果 - 每个声学特征单独展示"""
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        features = result['overall_features']
        
        f0_features = features.get('f0', {})
        if f0_features:
            self.create_feature_chart(
                "基频特征 (F0)",
                {
                    'Mean F0': f0_features.get('mean_f0_mean', 0),
                    'F0 Range': f0_features.get('range_f0_mean', 0),
                    'F0 Std': f0_features.get('std_f0_mean', 0)
                },
                'Hz',
                '#3498db'
            )
        
        spectral_features = features.get('spectral', {})
        if spectral_features:
            self.create_feature_chart(
                "频谱特征",
                {
                    'Spectral Centroid': spectral_features.get('spectral_centroid_mean_mean', 0),
                    'Spectral Bandwidth': spectral_features.get('spectral_bandwidth_mean_mean', 0)
                },
                'Hz',
                '#2980b9'
            )
        
        energy_features = features.get('energy', {})
        if energy_features:
            self.create_feature_chart(
                "能量特征",
                {
                    'RMS Energy': energy_features.get('rms_mean_mean', 0),
                    'Dynamic Range': energy_features.get('dynamic_range_mean', 0)
                },
                '',
                '#1abc9c'
            )
        
        quality_features = features.get('quality', {})
        if quality_features:
            self.create_feature_chart(
                "语音质量特征",
                {
                    'HNR': quality_features.get('hnr_mean', 0),
                    'Jitter': quality_features.get('jitter_mean', 0),
                    'Shimmer': quality_features.get('shimmer_mean', 0)
                },
                'dB/%',
                '#16a085'
            )
    
    def create_feature_chart(self, title, features_dict, unit, color):
        """创建单个声学特征图表"""
        group_box = QGroupBox(title)
        group_box.setMinimumHeight(300)
        group_layout = QVBoxLayout()
        
        canvas = MplCanvas(self, width=10, height=5)
        canvas.axes.clear()
        
        feature_names = list(features_dict.keys())
        feature_values = list(features_dict.values())
        
        max_val = max(feature_values) if max(feature_values) > 0 else 1
        normalized_values = [v / max_val for v in feature_values]
        
        x_pos = range(len(feature_names))
        bars = canvas.axes.bar(x_pos, normalized_values, color=color, alpha=0.8, width=0.6)
        
        canvas.axes.set_xticks(x_pos)
        canvas.axes.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=11)
        canvas.axes.set_ylabel('归一化值', fontsize=12)
        canvas.axes.set_title(title, fontsize=14, fontweight='bold', pad=15)
        canvas.axes.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, feature_values)):
            label = f'{val:.1f} {unit}' if unit else f'{val:.1f}'
            canvas.axes.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                label,
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                rotation=45
            )
        
        canvas.fig.tight_layout()
        canvas.draw()
        
        group_layout.addWidget(canvas)
        group_box.setLayout(group_layout)
        self.scroll_layout.addWidget(group_box)
    
    def export_result(self):
        if not self.current_result or not self.analyzer:
            self.parent.statusBar().showMessage("没有可导出的结果")
            return
        
        log_gui_operation(app_logger, "开始导出结果")
        
        output_file, _ = QFileDialog.getSaveFileName(
            self, "导出分析结果", ".", "JSON文件 (*.json)"
        )
        if output_file:
            try:
                self.analyzer.export_results(self.current_result, output_file)
                self.parent.statusBar().showMessage(f"结果已导出: {output_file}")
                log_gui_operation(app_logger, "导出结果成功", f"文件: {output_file}")
            except Exception as e:
                log_error(app_logger, e, "导出分析结果")
                self.parent.statusBar().showMessage(f"导出失败: {str(e)}")
    
    def analysis_error(self, error_msg):
        log_error(app_logger, Exception(error_msg), "人声分析")
        self.result_text.append(f"错误: {error_msg}")
        self.analyze_button.setEnabled(True)
        self.parent.statusBar().showMessage(f"分析失败: {error_msg}")
    
    def show_feature_help(self):
        """显示声学特征说明对话框"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("声学特征说明")
        help_dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        tabs = QTabWidget()
        
        # 基频特征
        f0_tab = QWidget()
        f0_layout = QVBoxLayout(f0_tab)
        f0_text = QTextEdit()
        f0_text.setReadOnly(True)
        f0_text.setHtml("""
        <h2>基频特征 (F0)</h2>
        <p>基频是声音的基本频率，决定了声音的音调高低。</p>
        <ul>
        <li><strong>平均基频</strong>: 说话人声音的平均音调高度。男性约 85-180 Hz，女性约 165-255 Hz。</li>
        <li><strong>基频范围</strong>: 音调变化的范围（最大值-最小值），值越大表示声音表现力越强。</li>
        <li><strong>基频标准差</strong>: 音调变化的离散程度，值越大表示音调起伏越大。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>性别识别：男性基频较低，女性基频较高</li>
        <li>情感分析：音调变化大通常表示激动或情绪化</li>
        </ul>
        """)
        f0_layout.addWidget(f0_text)
        tabs.addTab(f0_tab, "基频特征")
        
        # 频谱特征
        spectral_tab = QWidget()
        spectral_layout = QVBoxLayout(spectral_tab)
        spectral_text = QTextEdit()
        spectral_text.setReadOnly(True)
        spectral_text.setHtml("""
        <h2>频谱特征</h2>
        <p>频谱特征描述了声音在不同频率上的能量分布。</p>
        <ul>
        <li><strong>频谱质心</strong>: 频谱能量集中的频率位置，类似于声音的"明亮度"，值越高声音越明亮。</li>
        <li><strong>频谱带宽</strong>: 频谱能量分布的宽度，值越大表示声音包含的频率成分越丰富。</li>
        <li><strong>频谱滚降点</strong>: 能量降至85%时的频率，反映声音的低频特性。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>语音识别：用于特征提取</li>
        <li>说话人识别：建立声纹特征</li>
        </ul>
        """)
        spectral_layout.addWidget(spectral_text)
        tabs.addTab(spectral_tab, "频谱特征")
        
        # 能量特征
        energy_tab = QWidget()
        energy_layout = QVBoxLayout(energy_tab)
        energy_text = QTextEdit()
        energy_text.setReadOnly(True)
        energy_text.setHtml("""
        <h2>能量特征</h2>
        <p>能量特征描述了声音的强度和响度特性。</p>
        <ul>
        <li><strong>RMS能量</strong>: 声音的平均强度（归一化），值越大声音越响。</li>
        <li><strong>能量动态范围</strong>: 声音强度变化的范围，值越大表示音量变化越明显。</li>
        <li><strong>总能量</strong>: 整个音频的总能量值。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>语音活动检测：判断是否有语音</li>
        <li>情感分析：能量变化反映情绪强度</li>
        </ul>
        """)
        energy_layout.addWidget(energy_text)
        tabs.addTab(energy_tab, "能量特征")
        
        # 语音质量特征
        quality_tab = QWidget()
        quality_layout = QVBoxLayout(quality_tab)
        quality_text = QTextEdit()
        quality_text.setReadOnly(True)
        quality_text.setHtml("""
        <h2>语音质量特征</h2>
        <p>语音质量特征评估声音的清晰度和稳定性。</p>
        <ul>
        <li><strong>HNR (谐波噪声比)</strong>: 声音的"纯净度"，值越高声音越清晰、噪音越少。正常 > 10 dB。</li>
        <li><strong>Jitter (抖动)</strong>: 基频的周期变化，值越小音调越稳定。正常 < 1%。</li>
        <li><strong>Shimmer (闪烁)</strong>: 振幅的周期变化，值越小声音越平滑。正常 < 3%。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>嗓音健康评估：检测声带疾病</li>
        <li>语音合成：评估合成语音质量</li>
        </ul>
        """)
        quality_layout.addWidget(quality_text)
        tabs.addTab(quality_tab, "语音质量")
        
        # MFCC特征
        mfcc_tab = QWidget()
        mfcc_layout = QVBoxLayout(mfcc_tab)
        mfcc_text = QTextEdit()
        mfcc_text.setReadOnly(True)
        mfcc_text.setHtml("""
        <h2>MFCC特征</h2>
        <p>MFCC (梅尔频率倒谱系数) 是语音识别中最常用的特征。</p>
        <ul>
        <li><strong>MFCC系数</strong>: 共13个系数，描述声音的频谱包络。</li>
        <li><strong>Delta MFCC</strong>: MFCC的一阶差分，反映动态变化。</li>
        <li><strong>Delta-Delta MFCC</strong>: MFCC的二阶差分，反映变化的加速度。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>语音识别：最核心的特征</li>
        <li>说话人识别：声纹特征</li>
        <li>情感识别：情绪特征提取</li>
        </ul>
        """)
        mfcc_layout.addWidget(mfcc_text)
        tabs.addTab(mfcc_tab, "MFCC特征")
        
        # 时域特征
        temporal_tab = QWidget()
        temporal_layout = QVBoxLayout(temporal_tab)
        temporal_text = QTextEdit()
        temporal_text.setReadOnly(True)
        temporal_text.setHtml("""
        <h2>时域特征</h2>
        <p>时域特征描述声音在时间维度上的特性。</p>
        <ul>
        <li><strong>过零率</strong>: 信号穿越零点的次数，区分清音和浊音。</li>
        <li><strong>有声比例</strong>: 有声部分占总时长的比例。</li>
        <li><strong>平均段长</strong>: 有声段的平均持续时间。</li>
        </ul>
        <h3>应用场景</h3>
        <ul>
        <li>语速分析：计算说话速度</li>
        <li>语音分割：区分语音和静音</li>
        </ul>
        """)
        temporal_layout.addWidget(temporal_text)
        tabs.addTab(temporal_tab, "时域特征")
        
        layout.addWidget(tabs)
        
        close_button = QPushButton("关闭")
        close_button.clicked.connect(help_dialog.close)
        layout.addWidget(close_button)
        
        help_dialog.setLayout(layout)
        help_dialog.exec_()
    
    def update_progress(self, progress_info):
        """更新进度条和控制台输出 - PowerShell风格"""
        stage = progress_info.get('stage', '')
        progress = progress_info.get('progress', 0)
        message = progress_info.get('message', '')
        current = progress_info.get('current', 0)
        total = progress_info.get('total', 0)
        
        self.progress_bar.setValue(progress)
        
        # PowerShell风格进度条
        bar_width = 50
        filled = int(progress * bar_width / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        console_line = f"[{bar}] {progress:3d}%  {stage}"
        if message:
            console_line += f" - {message}"
        if current > 0 and total > 0:
            console_line += f" ({current}/{total})"
        
        self.console_output.append(console_line)
        # 滚动到最新行
        cursor = self.console_output.textCursor()
        cursor.movePosition(cursor.End)
        self.console_output.setTextCursor(cursor)
    
    def clear_console(self):
        """清空控制台"""
        self.console_output.clear()


class SpeakerTimelineTab(QWidget):
    """说话人时间线分析标签页 (XVector, 25人模型)"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)

        # 按钮
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("开始说话人分析 (XVector)")
        self.analyze_btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.analyze_btn)
        self.layout.addLayout(btn_layout)

        # 模型信息
        self.model_label = QLabel("模型: XVector | 25说话人 | 准确率99.97% | RTX 4060 GPU")
        self.model_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.layout.addWidget(self.model_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        # 结果文本
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(120)
        self.layout.addWidget(QLabel("分析结果:"))
        self.layout.addWidget(self.result_text)

        # 图表
        self.canvas = MplCanvas(self, width=10, height=5)
        self.layout.addWidget(QLabel("说话人活动时间线:"))
        self.layout.addWidget(self.canvas)

    def start_analysis(self):
        if not self.parent.current_audio:
            self.parent.statusBar().showMessage("请先在'音频导入'中加载音频文件")
            return

        self.analyze_btn.setEnabled(False)
        self.progress.setValue(10)
        self.result_text.clear()
        self.result_text.append("正在加载XVector模型并提取声纹特征...")

        def task():
            from speaker_timeline import analyze_speakers
            return analyze_speakers(self.parent.current_audio)

        self.worker = WorkerThread(task)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, result):
        self.progress.setValue(100)
        r = result

        self.result_text.clear()
        nv = r.get('voice_segments', r['num_segments'])
        ns = r.get('silent_segments', 0)
        self.result_text.append("=" * 55)
        self.result_text.append(f"  XVector 说话人分析结果 (25人模型)")
        self.result_text.append(f"  片段: {r['seg_dur']}s x{r['num_segments']} (有声{nv}/静音{ns})")
        self.result_text.append("=" * 55)
        self.result_text.append(f"  音频时长: {r['total_duration']:.1f}秒")
        self.result_text.append(f"  检测到说话人: {r['speaker_count']}人")
        self.result_text.append(f"  轮廓系数: {r['silhouette_score']}")
        self.result_text.append("")

        if r['speaker_count'] == 0:
            self.result_text.append("  (无足够有声段用于分析)")
        else:
            for spk in sorted(r['speakers'], key=lambda x: -x['percentage']):
                self.result_text.append(
                    f"  [说话人{spk['id']+1}] "
                    f"{spk['gender_hint']} | {spk['segments']}片段 ({spk['percentage']:.1f}%) | "
                    f"F0={spk['avg_f0']:.0f}Hz"
                )
                for start, end in spk['active_ranges']:
                    self.result_text.append(f"    {start:.1f}s -- {end:.1f}s")
                self.result_text.append("")
        self.result_text.append("=" * 55)

        # 绘图
        self.canvas.axes.clear()
        speakers = sorted(r['speakers'], key=lambda x: x['id'])
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                  '#ffd92f', '#e5c494', '#b3b3b3']

        # 说话人
        y_idx = 0
        y_labels = []
        for spk in speakers:
            c = colors[spk['id'] % len(colors)]
            for start, end in spk['active_ranges']:
                self.canvas.axes.barh(y_idx, end - start, left=start, height=0.7,
                                       color=c, edgecolor='white', linewidth=0.5)
            y_labels.append((y_idx, f"说话人{spk['id']+1} ({spk['gender_hint']})"))
            y_idx += 1

        # 静音段
        sr = r.get('silent_ranges', [])
        if sr:
            for start, end in sr:
                self.canvas.axes.barh(y_idx, end - start, left=start, height=0.5,
                                       color='#cccccc', edgecolor='white', linewidth=0.5,
                                       alpha=0.5)
            y_labels.append((y_idx, '静音'))
            y_idx += 1

        self.canvas.axes.set_yticks([p[0] for p in y_labels])
        self.canvas.axes.set_yticklabels([p[1] for p in y_labels])
        self.canvas.axes.set_xlabel("时间 (秒)")
        self.canvas.axes.set_title(
            f"XVector 说话人活动时间线 | 共{r['speaker_count']}人 | 片段{r['seg_dur']}s")
        self.canvas.axes.set_xlim(0, r['total_duration'])
        self.canvas.axes.grid(axis='x', alpha=0.3)
        self.canvas.draw()

        self.analyze_btn.setEnabled(True)
        self.parent.statusBar().showMessage(
            f"分析完成: {r['speaker_count']}个说话人")

    def on_error(self, msg):
        self.result_text.append(f"错误: {msg}")
        self.analyze_btn.setEnabled(True)
        self.progress.setValue(0)


class EcapaTab(QWidget):
    """ECAPA-TDNN 高级声纹分析标签页"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)

        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("开始ECAPA声纹分析")
        self.analyze_btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.analyze_btn)
        self.layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        self.powered_label = QLabel("Powered by ECAPA-TDNN (VoxCeleb pretrained)")
        self.powered_label.setStyleSheet("color: #666; font-style: italic;")
        self.layout.addWidget(self.powered_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(120)
        self.layout.addWidget(QLabel("分析结果:"))
        self.layout.addWidget(self.result_text)

        self.canvas = MplCanvas(self, width=10, height=5)
        self.layout.addWidget(QLabel("说话人活动时间线:"))
        self.layout.addWidget(self.canvas)

    def start_analysis(self):
        if not self.parent.current_audio:
            self.parent.statusBar().showMessage("请先在'音频导入'中加载音频文件")
            return

        self.analyze_btn.setEnabled(False)
        self.progress.setValue(10)
        self.result_text.clear()
        self.result_text.append("正在加载ECAPA-TDNN模型并提取声纹特征...")

        self.worker = EcapaWorker(self.parent.current_audio)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, result):
        self.progress.setValue(100)
        r = result

        self.result_text.clear()
        nv = r.get('voice_segments', r['num_segments'])
        ns = r.get('silent_segments', 0)
        self.result_text.append("=" * 55)
        self.result_text.append(f"  ECAPA-TDNN 说话人分析结果")
        self.result_text.append(f"  片段: {r['seg_dur']}s x{r['num_segments']} (有声{nv}/静音{ns})")
        self.result_text.append("=" * 55)
        self.result_text.append(f"  音频时长: {r['total_duration']:.1f}秒")
        self.result_text.append(f"  检测到说话人: {r['speaker_count']}人")
        self.result_text.append(f"  轮廓系数: {r['silhouette_score']}")
        self.result_text.append(f"  Powered by ECAPA-TDNN (VoxCeleb pretrained)")
        self.result_text.append("")

        if r['speaker_count'] == 0:
            self.result_text.append("  (无足够有声段用于分析)")
        else:
            for spk in sorted(r['speakers'], key=lambda x: -x['percentage']):
                self.result_text.append(
                    f"  [说话人{spk['id']+1}] "
                    f"{spk['gender_hint']} | {spk['segments']}片段 ({spk['percentage']:.1f}%) | "
                    f"F0={spk['avg_f0']:.0f}Hz"
                )
                for start, end in spk['active_ranges']:
                    self.result_text.append(f"    {start:.1f}s -- {end:.1f}s")
                self.result_text.append("")
        self.result_text.append("=" * 55)

        # 绘图
        self.canvas.axes.clear()
        speakers = sorted(r['speakers'], key=lambda x: x['id'])
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                  '#ffd92f', '#e5c494', '#b3b3b3']

        y_idx = 0
        y_labels = []
        for spk in speakers:
            c = colors[spk['id'] % len(colors)]
            for start, end in spk['active_ranges']:
                self.canvas.axes.barh(y_idx, end - start, left=start, height=0.7,
                                       color=c, edgecolor='white', linewidth=0.5)
            y_labels.append((y_idx, f"说话人{spk['id']+1} ({spk['gender_hint']})"))
            y_idx += 1

        sr = r.get('silent_ranges', [])
        if sr:
            for start, end in sr:
                self.canvas.axes.barh(y_idx, end - start, left=start, height=0.5,
                                       color='#cccccc', edgecolor='white', linewidth=0.5,
                                       alpha=0.5)
            y_labels.append((y_idx, '静音'))
            y_idx += 1

        self.canvas.axes.set_yticks([p[0] for p in y_labels])
        self.canvas.axes.set_yticklabels([p[1] for p in y_labels])
        self.canvas.axes.set_xlabel("时间 (秒)")
        self.canvas.axes.set_title(
            f"ECAPA-TDNN 说话人活动时间线 | 共{r['speaker_count']}人 | 片段{r['seg_dur']}s")
        self.canvas.axes.set_xlim(0, r['total_duration'])
        self.canvas.axes.grid(axis='x', alpha=0.3)
        self.canvas.draw()

        self.analyze_btn.setEnabled(True)
        self.parent.statusBar().showMessage(
            f"ECAPA分析完成: {r['speaker_count']}个说话人")

    def on_error(self, msg):
        self.result_text.append(f"错误: {msg}")
        self.analyze_btn.setEnabled(True)
        self.progress.setValue(0)


class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoSense")
        self.setGeometry(100, 100, 1200, 900)
        
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'EchoSense.jpg')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        
        self.audio_tab = AudioTab(self)
        self.tabs.addTab(self.audio_tab, "音频导入")
        
        self.denoise_tab = DenoiseTab(self)
        self.tabs.addTab(self.denoise_tab, "音频降噪")
        
        self.gender_tab = GenderTab(self)
        self.tabs.addTab(self.gender_tab, "性别识别")
        
        self.voice_analysis_tab = VoiceAnalysisTab(self)
        self.tabs.addTab(self.voice_analysis_tab, "人声分析")

        self.speaker_tab = SpeakerTimelineTab(self)
        self.tabs.addTab(self.speaker_tab, "说话人分析(XVector)")

        self.ecapa_tab = EcapaTab(self)
        self.tabs.addTab(self.ecapa_tab, "高级声纹(ECAPA)")
        
        main_layout.addWidget(self.tabs)
        
        self.statusBar().showMessage("就绪")
        
        self.current_audio = None


if __name__ == "__main__":
    import traceback, atexit, time
    
    # 退出时写日志
    crash_log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crash_log.txt')
    def on_exit():
        with open(crash_log, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 程序退出\n")
    atexit.register(on_exit)
    
    # 全局异常捕获
    def global_excepthook(exc_type, exc_value, exc_tb):
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n[FATAL] 未捕获异常:\n{error_msg}", file=sys.stderr)
        with open(crash_log, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 崩溃:\n{error_msg}\n")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = global_excepthook
    
    # GPU 内存检查
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            used_mem = total_mem - free_mem
            print(f"[GPU] 显存状态: 已用 {used_mem:.1f}GB / 总计 {total_mem:.1f}GB (空闲 {free_mem:.1f}GB)")
            if free_mem < 1.5:
                print("[GPU] ⚠ 可用显存不足 1.5GB，部分模型可能加载失败！")
    except Exception as e:
        print(f"[GPU] 显存检查失败: {e}")
    
    print("[APP] 正在初始化界面...")
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        print("[APP] 主窗口创建完成，正在显示...")
        window.show()
        print("[APP] 窗口已显示，进入事件循环")
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[FATAL] 主窗口初始化失败:\n{error_msg}", file=sys.stderr)
        with open(crash_log, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化失败:\n{error_msg}\n")
        sys.exit(1)
    sys.exit(app.exec_())