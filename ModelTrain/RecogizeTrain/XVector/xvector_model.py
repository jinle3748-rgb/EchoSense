#!/usr/bin/env python3
"""
X-vector模型定义

X-vector是一种先进的声纹识别模型，基于TDNN（时间延迟神经网络），
能够提取固定长度的说话人嵌入向量，用于说话人识别和验证。
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def create_xvector_model(input_shape, num_speakers):
    """
    创建X-vector模型
    
    参数:
        input_shape: 输入特征的形状 (时间步长, 特征维度)
        num_speakers: 说话人数量
    
    返回:
        model: 完整的X-vector模型
        embedding_model: 用于提取嵌入的模型
    """
    # 输入层
    inputs = layers.Input(shape=input_shape, name='input')
    
    # 第一层卷积
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    
    # 第二层卷积
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                     name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # 第三层卷积
    x = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                     name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.Dropout(0.4, name='dropout3')(x)
    
    # 第四层卷积
    x = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                     name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    x = layers.Dropout(0.4, name='dropout4')(x)
    
    # 统计池化层
    # 计算均值和标准差
    mean = layers.GlobalAveragePooling1D(name='mean_pool')(x)
    std = layers.GlobalMaxPooling1D(name='std_pool')(x)
    
    # 拼接均值和标准差
    x = layers.Concatenate(name='stats_pool')([mean, std])
    
    # 嵌入层
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0001), name='embed')(x)
    x = layers.BatchNormalization(name='bn_embed')(x)
    x = layers.ReLU(name='relu_embed')(x)
    x = layers.Dropout(0.5, name='dropout_embed')(x)
    
    # 说话人嵌入向量
    embedding = x
    
    # 分类层
    outputs = layers.Dense(num_speakers, activation='softmax', name='output')(x)
    
    # 创建完整模型
    model = models.Model(inputs=inputs, outputs=outputs, name='xvector_model')
    
    # 创建嵌入提取模型
    embedding_model = models.Model(inputs=inputs, outputs=embedding, name='xvector_embedding')
    
    return model, embedding_model


def compile_xvector_model(model, learning_rate=0.001):
    """
    编译X-vector模型
    
    参数:
        model: X-vector模型
        learning_rate: 学习率
    
    返回:
        model: 编译后的模型
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    """
    获取训练回调
    
    返回:
        callbacks: 回调列表
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='RecogizeTrain/XVector/models/xvector_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=8,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='RecogizeTrain/XVector/logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    return callbacks
