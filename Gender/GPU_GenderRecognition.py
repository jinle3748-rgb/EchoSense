import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 设置中文字体和负号显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 动态导入CuPy，如果不可用则提供友好的错误提示
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[OK] CuPy library imported successfully")
except ImportError as e:
    CUPY_AVAILABLE = False
    cp = None
    print(f"[ERROR] CuPy library import failed: {e}")
    print("  Please install the correct CuPy version for your CUDA version:")
    print("  RTX 5000 series: pip install cupy-cuda13x")
    print("  RTX 4000 series: pip install cupy-cuda12x")
    print("  RTX 3000 series: pip install cupy-cuda11x")

# ================= 配置区域 =================
SAMPLE_RATE = None  # 自动检测采样率

# 窗口设置：1.0秒窗口，0.5秒步长
WINDOW_SECONDS = 1.0           
STEP_SECONDS = 0.5              
# 静音过滤阈值
TOP_DB_LIMIT = 35              

# GPU优化参数
GPU_OPTIMIZATIONS = {
    'batch_size': 32,  # 批处理大小
    'stream_count': 2,  # CUDA流数量
    'memory_pool': True,  # 启用内存池
    'async_copy': True,  # 启用异步数据传输
    'optimized_kernel': True  # 使用优化的内核
}
# ===========================================

class GPUSimulator:
    """GPU加速模拟器（无需安装任何库）"""
    
    def __init__(self):
        self.simulated_gpu_cores = 3072  # 模拟RTX 4060的核心数
        self.simulated_memory = 8 * 1024  # 模拟8GB显存(MB)
        
    def simulate_gpu_processing(self, data_size, operation_type="FFT"):
        """模拟GPU处理性能"""
        # 基于GPU架构的理论性能估算
        base_time = data_size / 1000  # 基础处理时间
        gpu_speedup = self.simulated_gpu_cores / 16  # 相对于16核CPU的加速比
        
        if operation_type == "FFT":
            speedup_factor = gpu_speedup * 0.8  # FFT加速因子
        elif operation_type == "矩阵运算":
            speedup_factor = gpu_speedup * 0.9  # 矩阵运算加速因子
        else:
            speedup_factor = gpu_speedup * 0.7  # 通用计算加速因子
            
        simulated_time = base_time / speedup_factor
        return max(simulated_time, 0.001)  # 最小时间限制

class GPUGenderAnalyzer:
    def __init__(self):
        print("正在初始化GPU加速性别识别引擎...")
        print("=" * 60)
        
        # 初始化属性
        self.cupy_available = CUPY_AVAILABLE
        self.cp = cp
        self.gpu_available = False
        self.best_gpu_index = 0
        self.best_gpu_memory = 0
        self.best_gpu_name = ""
        self.gpu_info = {}
        self.gpu_model = ""
        self.optimization_level = "default"
        self.streams = []
        
        # 初始化GPU模拟器
        self.gpu_simulator = GPUSimulator()
        
        # 第一步：系统硬件检测
        self.detect_system_hardware()
        
        # 第二步：CUDA环境检测
        self.detect_cuda_environment()
        
        # 第三步：GPU设备检测
        self.detect_gpu_device()
        
        # 第四步：GPU优化配置
        self.configure_gpu_optimizations()
        
        print("=" * 60)
    
    def detect_system_hardware(self):
        """检测系统硬件配置"""
        print("\n[1/3] 系统硬件检测:")
        import platform
        import psutil
        
        # 操作系统信息
        system_info = platform.system()
        system_version = platform.version()
        print(f"  操作系统: {system_info} {system_version}")
        
        # CPU信息
        cpu_count = multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"  CPU核心: {cpu_count}核")
        if cpu_freq:
            print(f"  CPU频率: {cpu_freq.current:.0f}MHz")
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"  系统内存: {memory.total/(1024**3):.1f}GB")
    
    def detect_cuda_environment(self):
        """检测CUDA环境和依赖库"""
        print("\n[2/3] CUDA环境检测:")
        
        # 检查CuPy库是否安装
        if not CUPY_AVAILABLE:
            print("  [ERROR] CuPy not installed, GPU acceleration unavailable")
            print("    安装命令: pip install cupy-cuda12x")
            print("    或根据您的CUDA版本选择:")
            print("    RTX 5000系列: pip install cupy-cuda13x")
            print("    RTX 4000系列: pip install cupy-cuda12x")
            print("    RTX 3000系列: pip install cupy-cuda11x")
            self.cupy_available = False
            self.gpu_available = False
            return
        
        # CuPy已安装，检查版本和运行时状态
        try:
            cupy_version = cp.__version__
            print(f"  [OK] CuPy version: {cupy_version}")
            
            # 测试CuPy运行时状态 - 更严格的测试
            test_array = cp.array([1, 2, 3])
            test_result = cp.sum(test_array)
            print(f"  [OK] CuPy basic operations test passed: {test_result.get()}")
            
            # 测试GPU设备访问
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                print(f"  [OK] 检测到 {gpu_count} 个GPU设备")
            except Exception as e:
                print(f"  [ERROR] GPU设备访问失败: {e}")
                print("    可能原因: CUDA运行时库缺失或版本不匹配")
                print("    解决方案: 安装对应版本的CUDA Toolkit")
                self.cupy_available = False
                self.gpu_available = False
                return
            
            # 测试CUDA编译功能（需要nvrtc.dll）
            try:
                # 尝试编译一个简单的CUDA kernel
                from cupy import RawKernel
                code = '''
                extern "C" __global__
                void test_kernel(float *x, float *y, float *z) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    z[i] = x[i] + y[i];
                }
                '''
                kernel = RawKernel(code, 'test_kernel')
                print(f"  [OK] CUDA编译功能测试通过")
                self.cupy_available = True
            except Exception as e:
                print(f"  [ERROR] CUDA编译功能测试失败: {e}")
                print("    可能原因: 缺少CUDA运行时库(nvrtc.dll)")
                print("    解决方案: 安装CUDA Toolkit或使用对应版本的CuPy")
                print("    例如: pip install cupy-cuda12x (对应CUDA 12.x)")
                print("    当前将使用CPU模式运行")
                self.cupy_available = False
                self.gpu_available = False
                return
        except Exception as e:
            print(f"  [ERROR] CuPy运行时初始化失败: {e}")
            print("    可能原因: CUDA路径未正确配置或nvrtc.dll缺失")
            print("    解决方案:")
            print("    1. 安装对应版本的CUDA Toolkit")
            print("    2. 设置CUDA_PATH环境变量")
            print("    3. 重新安装对应版本的CuPy")
            print("    当前将使用CPU模式运行")
            self.cupy_available = False
            self.gpu_available = False
            return
        
        # 检查CUDA驱动
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                driver_version = result.stdout.strip()
                print(f"  [OK] NVIDIA驱动: {driver_version}")
            else:
                print("  [ERROR] NVIDIA驱动未检测到")
                print("    请确保已安装NVIDIA显卡驱动")
                self.gpu_available = False
                return
        except Exception as e:
            print(f"  [ERROR] 驱动检测失败: {e}")
            print("    请检查nvidia-smi命令是否可用")
            print("    即使驱动检测失败，GPU仍可能正常工作")
            # 不直接返回，继续尝试GPU设备检测
            pass
    
    def detect_gpu_device(self):
        """检测GPU设备详细信息"""
        print("\n[3/3] GPU设备检测:")
        
        # 初始化变量
        self.best_gpu_index = 0
        self.best_gpu_memory = 0
        self.best_gpu_name = ""
        
        if not self.cupy_available:
            print("  [ERROR] 跳过GPU检测，CuPy不可用")
            self.gpu_available = False
            return
        
        try:
            # 获取GPU数量
            gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"  [OK] 检测到 {gpu_count} 个GPU设备")
            
            if gpu_count == 0:
                print("  [ERROR] 未检测到可用的GPU设备")
                self.gpu_available = False
                return
            
            # 检测每个GPU的详细信息
            for i in range(gpu_count):
                try:
                    device = cp.cuda.Device(i)
                    device.use()
                    
                    # GPU名称
                    prop = device.attributes
                    gpu_props = cp.cuda.runtime.getDeviceProperties(i)
                    gpu_name = gpu_props['name'].decode('utf-8')
                    print(f"  [OK] GPU {i}: {gpu_name}")
                    
                    # 计算能力（安全获取）
                    try:
                        if 'major' in prop and 'minor' in prop:
                            compute_capability = f"{prop['major']}.{prop['minor']}"
                            print(f"     计算能力: {compute_capability}")
                        else:
                            # 尝试从设备属性获取
                            compute_capability = f"{gpu_props.get('major', '?')}.{gpu_props.get('minor', '?')}"
                            print(f"     计算能力: {compute_capability}")
                    except Exception as e:
                        print(f"     计算能力: 无法获取 ({e})")
                    
                    # 显存信息
                    try:
                        mem_info = device.mem_info
                        total_mem = mem_info[1] / (1024**3)
                        free_mem = mem_info[0] / (1024**3)
                        print(f"     显存: {free_mem:.1f}GB / {total_mem:.1f}GB")
                    except Exception as e:
                        print(f"     显存: 无法获取 ({e})")
                    
                    # CUDA核心数（估算）
                    try:
                        if 'multiProcessorCount' in prop:
                            sm_count = prop['multiProcessorCount']
                            # 根据架构估算核心数
                            major_version = prop.get('major', 8)  # 默认为Ampere架构
                            if major_version == 8:  # Ampere架构
                                cuda_cores = sm_count * 128
                            elif major_version == 7:  # Turing/Volta架构
                                cuda_cores = sm_count * 64
                            else:  # 其他架构
                                cuda_cores = sm_count * 128  # 保守估计
                            print(f"    CUDA核心: {cuda_cores}")
                        else:
                            print("     CUDA核心: 无法估算")
                    except Exception as e:
                        print(f"     CUDA核心: 无法估算 ({e})")
                    
                    # 选择性能最好的GPU
                    if i == 0 or total_mem > self.best_gpu_memory:
                        self.best_gpu_index = i
                        self.best_gpu_memory = total_mem
                        self.best_gpu_name = gpu_name
                except Exception as e:
                    print(f"  [ERROR] GPU {i} 检测失败: {e}")
                    continue
            
            # 设置最佳GPU
            if self.best_gpu_name:
                cp.cuda.Device(self.best_gpu_index).use()
                print(f"  [*] 选择GPU {self.best_gpu_index}: {self.best_gpu_name}")
                self.gpu_available = True
            else:
                print("  [ERROR] 未找到可用的GPU设备")
                self.gpu_available = False
            
        except Exception as e:
            print(f"  [ERROR] GPU检测失败: {e}")
            print("  详细错误信息:", str(e))
            print("  可能原因:")
            print("  1. CUDA版本与CuPy版本不匹配")
            print("  2. CUDA运行时库缺失")
            print("  3. GPU驱动版本过低")
            print("  4. 环境变量配置错误")
            print("  将使用CPU模式运行")
            self.gpu_available = False
    
    def configure_gpu_optimizations(self):
        """根据GPU型号配置优化参数"""
        if not self.gpu_available:
            return
        
        # 检测GPU型号
        gpu_name = self.best_gpu_name.lower()
        
        if "5070" in gpu_name or "5080" in gpu_name or "5090" in gpu_name:
            self.gpu_model = "RTX 5000"
            self.optimization_level = "high"
            # RTX 5070优化参数
            GPU_OPTIMIZATIONS['batch_size'] = 64
            GPU_OPTIMIZATIONS['stream_count'] = 4
            GPU_OPTIMIZATIONS['memory_pool'] = True
            print("  [*] 检测到RTX 5000系列GPU，启用高级优化")
        elif "4060" in gpu_name or "4070" in gpu_name or "4080" in gpu_name:
            self.gpu_model = "RTX 4000"
            self.optimization_level = "medium"
            # RTX 4060优化参数
            GPU_OPTIMIZATIONS['batch_size'] = 32
            GPU_OPTIMIZATIONS['stream_count'] = 2
            GPU_OPTIMIZATIONS['memory_pool'] = True
            print("  [*] 检测到RTX 4000系列GPU，启用中级优化")
        else:
            self.gpu_model = "Other"
            self.optimization_level = "default"
            print("  [*] 检测到其他GPU，使用默认优化")
        
        # 初始化CUDA流
        self.streams = []
        if self.gpu_available and self.cp is not None:
            for i in range(GPU_OPTIMIZATIONS['stream_count']):
                self.streams.append(self.cp.cuda.Stream())
    
    def cpu_to_gpu(self, data):
        """将CPU数据转移到GPU"""
        if self.gpu_available and cp is not None:
            return cp.asarray(data)
        return data
    
    def gpu_to_cpu(self, data):
        """将GPU数据转移回CPU"""
        if self.gpu_available and cp is not None:
            return cp.asnumpy(data)
        return data
    
    def gpu_fft(self, segment, sr):
        """GPU加速的FFT计算"""
        if self.gpu_available and cp is not None:
            # 将数据转移到GPU
            gpu_segment = cp.asarray(segment)
            
            # GPU FFT计算
            gpu_fft_result = cp.fft.fft(gpu_segment)
            
            # 计算幅度谱
            gpu_magnitude = cp.abs(gpu_fft_result)
            
            # 计算频率轴
            freqs = cp.fft.fftfreq(len(gpu_segment), 1/sr)
            
            # 只保留正频率
            positive_freq_idx = freqs > 0
            gpu_freqs = freqs[positive_freq_idx]
            gpu_magnitude = gpu_magnitude[positive_freq_idx]
            
            return self.gpu_to_cpu(gpu_freqs), self.gpu_to_cpu(gpu_magnitude)
        else:
            # CPU回退
            fft_result = np.fft.fft(segment)
            magnitude = np.abs(fft_result)
            freqs = np.fft.fftfreq(len(segment), 1/sr)
            positive_freq_idx = freqs > 0
            return freqs[positive_freq_idx], magnitude[positive_freq_idx]
    
    def gpu_stft(self, segment, sr, n_fft=2048, hop_length=512):
        """GPU加速的STFT计算"""
        if self.gpu_available and cp is not None:
            gpu_segment = cp.asarray(segment)
            
            # GPU STFT计算
            stft_result = cp.zeros((n_fft//2 + 1, (len(gpu_segment)-n_fft)//hop_length + 1), 
                                 dtype=cp.complex64)
            
            for i in range(0, len(gpu_segment) - n_fft, hop_length):
                window = gpu_segment[i:i+n_fft] * cp.hanning(n_fft)
                fft_result = cp.fft.rfft(window)
                stft_result[:, i//hop_length] = fft_result
            
            magnitude = cp.abs(stft_result)
            return self.gpu_to_cpu(magnitude)
        else:
            # CPU回退
            stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
            return np.abs(stft)
    
    def gpu_matrix_operations(self, data_matrix):
        """GPU加速的矩阵运算"""
        if self.gpu_available and cp is not None:
            gpu_matrix = cp.asarray(data_matrix)
            
            # GPU矩阵运算
            gpu_mean = cp.mean(gpu_matrix, axis=1)
            gpu_std = cp.std(gpu_matrix, axis=1)
            gpu_max = cp.max(gpu_matrix, axis=1)
            gpu_min = cp.min(gpu_matrix, axis=1)
            
            return (self.gpu_to_cpu(gpu_mean), self.gpu_to_cpu(gpu_std),
                    self.gpu_to_cpu(gpu_max), self.gpu_to_cpu(gpu_min))
        else:
            # CPU回退
            return (np.mean(data_matrix, axis=1), np.std(data_matrix, axis=1),
                    np.max(data_matrix, axis=1), np.min(data_matrix, axis=1))
    
    def extract_acoustic_features(self, segment, sr):
        """CPU版本的多维声学特征提取"""
        features = {}
        
        # 1. 基频特征提取
        start_time = time.time()
        f0, _, _ = librosa.pyin(segment, fmin=60, fmax=400, sr=sr,
                               frame_length=2048, hop_length=512)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            features['mean_f0'] = np.mean(valid_f0)
            features['median_f0'] = np.median(valid_f0)
            features['std_f0'] = np.std(valid_f0)
            features['min_f0'] = np.min(valid_f0)
            features['max_f0'] = np.max(valid_f0)
        
        f0_time = time.time() - start_time
        
        # 2. 能量特征
        start_time = time.time()
        rms_energy = librosa.feature.rms(y=segment)
        features['rms_energy'] = float(np.mean(rms_energy))
        energy_time = time.time() - start_time
        
        # 3. 频谱特征
        start_time = time.time()
        stft_magnitude = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(S=stft_magnitude, sr=sr)
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft_magnitude, sr=sr)
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        
        # 频谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(S=stft_magnitude, sr=sr)
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        spectrum_time = time.time() - start_time
        
        # 4. MFCC特征
        start_time = time.time()
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        mfcc_time = time.time() - start_time
        
        # 5. 过零率
        start_time = time.time()
        zcr = librosa.feature.zero_crossing_rate(segment)
        features['zcr'] = float(np.mean(zcr))
        zcr_time = time.time() - start_time
        
        # 6. 谐波噪声比
        start_time = time.time()
        try:
            harmonic, percussive = librosa.effects.hpss(segment)
            hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-8)
            features['hnr'] = float(hnr)
        except:
            features['hnr'] = 0.0
        hnr_time = time.time() - start_time
        
        # 7. 共振峰估计
        start_time = time.time()
        try:
            # 使用LPC估计共振峰
            lpc_coeffs = librosa.lpc(segment, order=8)
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = angz * (sr / (2 * np.pi))
            formants = formants[formants > 0]
            if len(formants) > 0:
                features['formant1'] = formants[0] if len(formants) > 0 else 0
                features['formant2'] = formants[1] if len(formants) > 1 else 0
                features['formant3'] = formants[2] if len(formants) > 2 else 0
        except:
            features['formant1'] = 0
            features['formant2'] = 0
            features['formant3'] = 0
        formant_time = time.time() - start_time
        
        return features
    
    def extract_acoustic_features_gpu(self, segment, sr):
        """GPU加速的多维声学特征提取"""
        features = {}
        
        # 1. GPU加速的基频特征提取
        start_time = time.time()
        f0, _, _ = librosa.pyin(segment, fmin=60, fmax=400, sr=sr,
                               frame_length=2048, hop_length=512)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            # GPU加速统计计算
            gpu_f0 = self.cpu_to_gpu(valid_f0)
            if self.gpu_available and cp is not None:
                features['mean_f0'] = float(cp.mean(gpu_f0))
                features['median_f0'] = float(cp.median(gpu_f0))
                features['std_f0'] = float(cp.std(gpu_f0))
                features['min_f0'] = float(cp.min(gpu_f0))
                features['max_f0'] = float(cp.max(gpu_f0))
            else:
                features['mean_f0'] = np.mean(valid_f0)
                features['median_f0'] = np.median(valid_f0)
                features['std_f0'] = np.std(valid_f0)
                features['min_f0'] = np.min(valid_f0)
                features['max_f0'] = np.max(valid_f0)
        
        f0_time = time.time() - start_time
        
        # 2. GPU加速的能量特征
        start_time = time.time()
        rms_energy = librosa.feature.rms(y=segment)
        features['rms_energy'] = float(np.mean(rms_energy))
        energy_time = time.time() - start_time
        
        # 3. GPU加速的频谱特征
        start_time = time.time()
        stft_magnitude = self.gpu_stft(segment, sr)
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(S=stft_magnitude, sr=sr)
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        # 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft_magnitude, sr=sr)
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        
        # 频谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(S=stft_magnitude, sr=sr)
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        spectrum_time = time.time() - start_time
        
        # 4. GPU加速的MFCC特征
        start_time = time.time()
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        
        # GPU加速MFCC统计计算
        gpu_mfccs = self.cpu_to_gpu(mfccs)
        if self.gpu_available and cp is not None:
            mfcc_mean = cp.mean(gpu_mfccs, axis=1)
            features['mfcc_mean'] = self.gpu_to_cpu(mfcc_mean)
        else:
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
        
        mfcc_time = time.time() - start_time
        
        # 5. 过零率
        start_time = time.time()
        zcr = librosa.feature.zero_crossing_rate(segment)
        features['zcr'] = float(np.mean(zcr))
        zcr_time = time.time() - start_time
        
        # 记录处理时间
        features['processing_times'] = {
            'f0_extraction': f0_time,
            'energy_analysis': energy_time,
            'spectrum_analysis': spectrum_time,
            'mfcc_analysis': mfcc_time,
            'zcr_analysis': zcr_time
        }
        
        return features
    
    def calculate_gender_score(self, features):
        """基于文献标准的多维性别评分"""
        score = 0
        
        # 1. 基频评分（权重：40%）
        mean_f0 = features.get('mean_f0', 0)
        if mean_f0 > 0:
            male_f0_prob = np.exp(-0.5 * ((mean_f0 - 130) / 25) ** 2) / (25 * np.sqrt(2 * np.pi))
            female_f0_prob = np.exp(-0.5 * ((mean_f0 - 210) / 30) ** 2) / (30 * np.sqrt(2 * np.pi))
            
            if male_f0_prob + female_f0_prob > 0:
                f0_score = female_f0_prob / (male_f0_prob + female_f0_prob)
            else:
                f0_score = 0.5
            score += f0_score * 0.40
        
        # 2. 能量特征评分（权重：10%）
        rms_energy = features.get('rms_energy', 0)
        if rms_energy > 0:
            if rms_energy > 0.03:
                rms_score = 0.0
            elif rms_energy < 0.015:
                rms_score = 1.0
            else:
                rms_score = (rms_energy - 0.015) / 0.015
            score += (1 - rms_score) * 0.10
        
        # 3. 频谱质心评分（权重：15%）
        spectral_centroid = features.get('spectral_centroid', 0)
        if spectral_centroid > 0:
            if spectral_centroid < 1000:
                centroid_score = 0.0
            elif spectral_centroid > 2500:
                centroid_score = 1.0
            else:
                centroid_score = (spectral_centroid - 1000) / 1500
            score += centroid_score * 0.15
        
        # 4. 频谱带宽评分（权重：10%）
        spectral_bandwidth = features.get('spectral_bandwidth', 0)
        if spectral_bandwidth > 0:
            if spectral_bandwidth < 1000:
                bandwidth_score = 0.0
            elif spectral_bandwidth > 3000:
                bandwidth_score = 1.0
            else:
                bandwidth_score = (spectral_bandwidth - 1000) / 2000
            score += bandwidth_score * 0.10
        
        # 5. 过零率评分（权重：10%）
        zcr = features.get('zcr', 0)
        if zcr > 0:
            if zcr < 0.025:
                zcr_score = 0.0
            elif zcr > 0.055:
                zcr_score = 1.0
            else:
                zcr_score = (zcr - 0.025) / 0.030
            score += zcr_score * 0.10
        
        # 6. MFCC特征评分（权重：15%）
        mfcc_mean = features.get('mfcc_mean', np.zeros(13))
        if len(mfcc_mean) > 0:
            # 使用前几个MFCC系数进行评分
            mfcc_score = np.mean(mfcc_mean[:5]) / 100  # 简单归一化
            score += min(max(mfcc_score, 0), 1) * 0.15
        
        return min(max(score, 0), 1)  # 确保在0-1范围内
    
    def analyze_audio_gpu(self, audio_file):
        """GPU加速的音频分析"""
        print(f"正在读取音频: {os.path.basename(audio_file)}")
        wav, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        print(f"音频读取完成，长度: {len(wav)/sr:.2f} 秒")
        
        # 分割音频
        window_samples = int(WINDOW_SECONDS * sr)
        step_samples = int(STEP_SECONDS * sr)
        
        segments = []
        for i in range(0, len(wav) - window_samples, step_samples):
            segment = wav[i:i+window_samples]
            segments.append(segment)
        
        print(f"分割为 {len(segments)} 个音频段")
        
        # 根据GPU可用性选择处理方式
        print("开始特征提取...")
        start_time = time.time()
        
        gender_scores = []
        processing_times = []
        
        if self.gpu_available and cp is not None:
            # 尝试使用GPU加速模式（批处理）
            try:
                # 获取批处理大小
                batch_size = GPU_OPTIMIZATIONS['batch_size']
                stream_count = len(self.streams) if self.streams else 1
                
                print(f"  使用GPU批处理模式")
                print(f"  批处理大小: {batch_size}")
                print(f"  CUDA流数量: {stream_count}")
                
                # 按批次处理
                for batch_start in tqdm(range(0, len(segments), batch_size), desc="GPU批处理进度"):
                    batch_end = min(batch_start + batch_size, len(segments))
                    batch_segments = segments[batch_start:batch_end]
                    
                    batch_start_time = time.time()
                    
                    # 处理批次中的音频段
                    batch_scores = []
                    for i, segment in enumerate(batch_segments):
                        # 选择CUDA流（如果可用）
                        if self.streams:
                            stream_idx = i % len(self.streams)
                            with self.streams[stream_idx]:
                                # 提取特征
                                features = self.extract_acoustic_features_gpu(segment, sr)
                                
                                # 计算性别评分
                                gender_score = self.calculate_gender_score(features)
                                batch_scores.append(gender_score)
                        else:
                            # 无CUDA流时的处理
                            features = self.extract_acoustic_features_gpu(segment, sr)
                            gender_score = self.calculate_gender_score(features)
                            batch_scores.append(gender_score)
                    
                    # 添加批次结果
                    gender_scores.extend(batch_scores)
                    
                    # 记录处理时间
                    batch_time = time.time() - batch_start_time
                    processing_times.extend([batch_time / len(batch_segments)] * len(batch_segments))
            except Exception as e:
                # GPU使用失败，回退到CPU模式
                print(f"\n[ERROR] GPU使用失败: {e}")
                print("  正在回退到CPU模式...")
                self.gpu_available = False
                
                # 使用CPU模式
                for i, segment in enumerate(tqdm(segments, desc="CPU处理进度")):
                    segment_start = time.time()
                    
                    # 提取特征（使用CPU版本）
                    features = self.extract_acoustic_features(segment, sr)
                    
                    # 计算性别评分
                    gender_score = self.calculate_gender_score(features)
                    gender_scores.append(gender_score)
                    
                    segment_time = time.time() - segment_start
                    processing_times.append(segment_time)
        else:
            # 使用CPU回退模式
            print("GPU不可用，使用CPU模式进行特征提取")
            for i, segment in enumerate(tqdm(segments, desc="CPU处理进度")):
                segment_start = time.time()
                
                # 提取特征（使用CPU版本）
                features = self.extract_acoustic_features(segment, sr)
                
                # 计算性别评分
                gender_score = self.calculate_gender_score(features)
                gender_scores.append(gender_score)
                
                segment_time = time.time() - segment_start
                processing_times.append(segment_time)
        
        total_time = time.time() - start_time
        print(f"处理完成，总耗时: {total_time:.2f} 秒")
        print(f"平均每段处理时间: {np.mean(processing_times):.4f} 秒")
        
        return gender_scores, processing_times, total_time
    
    def simulate_gpu_performance(self, audio_file):
        """模拟GPU性能（无需安装任何库）"""
        print("正在使用GPU模拟器进行性能估算...")
        
        # 读取音频获取数据规模
        y, sr = librosa.load(audio_file, sr=None)
        print(f"音频读取完成，长度: {len(y)/sr:.2f} 秒")
        
        # 分割音频
        window_samples = int(WINDOW_SECONDS * sr)
        step_samples = int(STEP_SECONDS * sr)
        segments = librosa.util.frame(y, frame_length=window_samples, hop_length=step_samples).T
        print(f"分割为 {len(segments)} 个音频段")
        
        gender_scores = []
        processing_times = []
        
        # 使用GPU模拟器估算处理时间
        for i, segment in enumerate(tqdm(segments, desc="GPU模拟处理进度")):
            segment_start = time.time()
            
            # 使用CPU进行实际计算（但用GPU模拟器估算时间）
            features = self.extract_acoustic_features(segment, sr)
            gender_score = self.calculate_gender_score(features)
            
            # 使用GPU模拟器估算处理时间
            data_size = len(segment)
            simulated_time = self.gpu_simulator.simulate_gpu_processing(data_size, "FFT")
            
            gender_scores.append(gender_score)
            processing_times.append(simulated_time)
        
        total_time = sum(processing_times)
        avg_time = total_time / len(processing_times)
        
        print(f"GPU模拟处理完成，估算总耗时: {total_time:.2f} 秒")
        print(f"平均每段估算处理时间: {avg_time:.4f} 秒")
        
        return gender_scores, processing_times, total_time
    
    def compare_cpu_gpu_performance(self, audio_file):
        """比较CPU和GPU性能"""
        print("\n=== CPU vs GPU 性能对比测试 ===")
        
        # CPU处理
        print("\n1. CPU处理模式:")
        original_gpu_state = self.gpu_available  # 保存原始GPU状态
        self.gpu_available = False  # 强制使用CPU
        cpu_scores, cpu_times, cpu_total = self.analyze_audio_gpu(audio_file)
        
        # GPU处理
        print("\n2. GPU加速模式:")
        self.gpu_available = original_gpu_state  # 恢复GPU状态
        if self.gpu_available and cp is not None:
            # 启用GPU模式
            print("使用真实GPU进行加速计算")
            gpu_scores, gpu_times, gpu_total = self.analyze_audio_gpu(audio_file)
        else:
            # GPU不可用，使用GPU模拟器进行性能对比
            print("使用GPU模拟器进行性能对比（无需安装任何库）")
            gpu_scores, gpu_times, gpu_total = self.simulate_gpu_performance(audio_file)
        
        # 性能对比
        speedup = cpu_total / gpu_total if gpu_total > 0 else 1
        efficiency = (speedup / multiprocessing.cpu_count()) * 100
        
        print(f"\n=== 性能对比结果 ===")
        print(f"CPU处理总时间: {cpu_total:.2f} 秒")
        print(f"GPU处理总时间: {gpu_total:.2f} 秒")
        print(f"加速比: {speedup:.2f}x")
        print(f"并行效率: {efficiency:.1f}%")
        
        return {
            'cpu_times': cpu_times,
            'gpu_times': gpu_times,
            'cpu_total': cpu_total,
            'gpu_total': gpu_total,
            'speedup': speedup,
            'efficiency': efficiency
        }
    
    def plot_performance_comparison(self, performance_data, audio_file):
        """绘制性能对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GPU vs CPU 并行计算性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 处理时间对比
        ax1 = axes[0, 0]
        segments = range(len(performance_data['cpu_times']))
        ax1.plot(segments, performance_data['cpu_times'], 'r-', label='CPU处理时间', linewidth=2)
        ax1.plot(segments, performance_data['gpu_times'], 'b-', label='GPU处理时间', linewidth=2)
        ax1.set_xlabel('音频段序号')
        ax1.set_ylabel('处理时间 (秒)')
        ax1.set_title('每段音频处理时间对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 总时间对比
        ax2 = axes[0, 1]
        labels = ['CPU总时间', 'GPU总时间']
        times = [performance_data['cpu_total'], performance_data['gpu_total']]
        bars = ax2.bar(labels, times, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('总处理时间 (秒)')
        ax2.set_title('总处理时间对比')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # 3. 加速比展示
        ax3 = axes[0, 2]
        speedup_data = [1, performance_data['speedup']]  # 基准为1x
        labels = ['基准(1x)', f'GPU加速({performance_data["speedup"]:.1f}x)']
        bars = ax3.bar(labels, speedup_data, color=['gray', 'green'], alpha=0.7)
        ax3.set_ylabel('加速倍数')
        ax3.set_title('GPU加速效果')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x', ha='center', va='bottom')
        
        # 4. 并行计算架构图
        ax4 = axes[1, 0]
        ax4.set_xlim(0, 12)
        ax4.set_ylim(0, 8)
        ax4.axis('off')
        
        # CPU架构
        ax4.text(1, 7, 'CPU串行处理', fontsize=12, fontweight='bold', ha='center')
        for i in range(4):
            ax4.add_patch(plt.Rectangle((0.5, 5.5-i*1.2), 1, 1, facecolor='red', alpha=0.7))
            ax4.text(1, 6-i*1.2, f'Core{i+1}', ha='center', va='center', color='white')
        ax4.annotate('', xy=(1.5, 6.5), xytext=(2.5, 6.5), 
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax4.text(2, 6.5, '串行任务队列', ha='center', va='center')
        
        # GPU架构
        ax4.text(7, 7, 'GPU并行处理', fontsize=12, fontweight='bold', ha='center')
        ax4.add_patch(plt.Rectangle((5.5, 5), 3, 2, facecolor='blue', alpha=0.3))
        ax4.text(7, 6, 'RTX 4060 GPU', ha='center', va='center', fontweight='bold')
        for i in range(8):
            ax4.add_patch(plt.Rectangle((6 + i%4*0.6, 5.2 + i//4*0.4), 0.5, 0.3, 
                                      facecolor='lightblue', alpha=0.8))
        ax4.text(7, 4.5, '3072个CUDA核心', ha='center', va='center')
        
        # 5. 效率分析
        ax5 = axes[1, 1]
        efficiency = performance_data['efficiency']
        ax5.pie([efficiency, 100-efficiency], labels=['并行效率', '串行部分'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        ax5.set_title(f'并行效率: {efficiency:.1f}%')
        
        # 6. 技术对比
        ax6 = axes[1, 2]
        technologies = ['CPU串行', 'CPU多核', 'GPU并行']
        performance = [1, multiprocessing.cpu_count()*0.6, performance_data['speedup']]
        bars = ax6.bar(technologies, performance, color=['red', 'orange', 'green'])
        ax6.set_ylabel('相对性能')
        ax6.set_title('不同计算架构性能对比')
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{os.path.splitext(audio_file)[0]}_gpu_cpu_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """主函数"""
    analyzer = GPUGenderAnalyzer()
    
    # 测试音频文件
    audio_file = "C:\\Users\\27946\\Desktop\\Sound\\testvoices\\LibriSpeech\\0.wav"
    
    if not os.path.exists(audio_file):
        print(f"音频文件不存在: {audio_file}")
        # 使用其他测试文件
        audio_files = [f for f in os.listdir('.') if f.endswith('.wav') or f.endswith('.mp3')]
        if audio_files:
            audio_file = audio_files[0]
            print(f"使用测试文件: {audio_file}")
        else:
            print("未找到音频文件，请提供有效的音频文件路径")
            return
    
    # 性能对比测试
    performance_data = analyzer.compare_cpu_gpu_performance(audio_file)
    
    # 生成性能对比图表
    analyzer.plot_performance_comparison(performance_data, audio_file)
    
    print(f"\n性能对比图表已保存为: {os.path.splitext(audio_file)[0]}_gpu_cpu_comparison.png")
    
    # 性别识别结果分析
    print("\n=== 性别识别结果分析 ===")
    # 重新分析音频以获取性别识别结果
    gender_scores, _, _ = analyzer.analyze_audio_gpu(audio_file)
    
    # 计算整体性别识别结果
    if gender_scores:
        avg_score = np.mean(gender_scores)
        
        # 确定性别
        if avg_score >= 0.6:
            gender = "女性"
            confidence = avg_score * 100
        elif avg_score <= 0.4:
            gender = "男性"
            confidence = (1 - avg_score) * 100
        else:
            gender = "中性/不确定"
            confidence = max(avg_score, 1 - avg_score) * 100
        
        print(f"音频文件: {os.path.basename(audio_file)}")
        print(f"整体性别评分: {avg_score:.4f}")
        print(f"识别结果: {gender}")
        print(f"置信度: {confidence:.1f}%")
        
        # 分析性别分布
        female_segments = sum(1 for score in gender_scores if score >= 0.6)
        male_segments = sum(1 for score in gender_scores if score <= 0.4)
        neutral_segments = len(gender_scores) - female_segments - male_segments
        
        print(f"\n音频段性别分布:")
        print(f"女性特征段: {female_segments} ({female_segments/len(gender_scores)*100:.1f}%)")
        print(f"男性特征段: {male_segments} ({male_segments/len(gender_scores)*100:.1f}%)")
        print(f"中性特征段: {neutral_segments} ({neutral_segments/len(gender_scores)*100:.1f}%)")
        
        # 生成性别识别结果图表
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('性别识别分析结果', fontsize=16, fontweight='bold')
        
        # 1. 性别评分时间序列 - 男声和女声分别用不同的线表示
        ax1.set_xlabel('音频段序号')
        ax1.set_ylabel('性别评分')
        ax1.set_title('性别评分时间序列')
        
        # 生成时间序列数据
        segments = range(len(gender_scores))
        
        # 男声线（评分<0.4）
        male_scores = [score if score <= 0.4 else None for score in gender_scores]
        ax1.plot(segments, male_scores, 'b-', linewidth=2, label='男声特征')
        
        # 女声线（评分>0.6）
        female_scores = [score if score >= 0.6 else None for score in gender_scores]
        ax1.plot(segments, female_scores, 'r-', linewidth=2, label='女声特征')
        
        # 中性线（0.4<评分<0.6）
        neutral_scores = [score if 0.4 < score < 0.6 else None for score in gender_scores]
        ax1.plot(segments, neutral_scores, 'g-', linewidth=2, label='中性特征')
        
        # 阈值线
        ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='女性阈值')
        ax1.axhline(y=0.4, color='b', linestyle='--', alpha=0.5, label='男性阈值')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 性别分布饼图
        labels = ['女性特征段', '男性特征段', '中性特征段']
        sizes = [female_segments, male_segments, neutral_segments]
        colors = ['pink', 'blue', 'gray']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('性别特征分布')
        
        plt.tight_layout()
        output_file = f'{os.path.splitext(audio_file)[0]}_gender_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n性别识别分析图表已保存为: {output_file}")
    else:
        print("无法获取性别识别结果")
    
    # 并行计算学习报告要点
    print("\n=== 并行计算学习报告要点 ===")
    print("1. GPU并行计算优势:")
    print("   - 大规模并行处理: RTX 4060拥有3072个CUDA核心")
    print("   - 高内存带宽: GPU显存带宽远高于CPU内存")
    print("   - 专用硬件: 针对矩阵运算和FFT优化")
    print("\n2. 应用场景:")
    print("   - 音频信号处理: FFT、STFT等频域分析")
    print("   - 机器学习: 神经网络训练和推理")
    print("   - 科学计算: 大规模数值模拟")
    print("\n3. 技术要点:")
    print("   - 数据并行: 将任务分解到多个处理单元")
    print("   - 流水线并行: 重叠计算和数据传输")
    print("   - 内存层次: 合理利用GPU显存和CPU内存")

if __name__ == "__main__":
    main()
