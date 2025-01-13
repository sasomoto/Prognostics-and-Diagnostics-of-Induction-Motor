class FaultAnalysisSystem:
    def __init__(self, sequence_length=50, prediction_horizon=10, threshold_warning=0.7, model_save_path=None, batch_size=128):
        # Initialization
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.threshold_warning = threshold_warning
        self.feature_scaler = StandardScaler()
        self.rul_scaler = MinMaxScaler()
        self.num_classes = 4
        self.class_names = ['Healthy', 'R0S25', 'R0S50', 'R0S75']
        self.model_save_path = model_save_path or 'models/best_fault_model.keras'
        self.batch_size = batch_size

        # GPU memory management
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1800)]
                )
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
