    def data_generator(self, file_paths, batch_size=None):
        """Memory-efficient data generator."""
        batch_size = batch_size or self.batch_size
        chunk_size = min(10000, batch_size * 50)

        # Fit scalers on a sample file
        sample_file = file_paths[0]
        df_sample = pd.read_csv(sample_file, nrows=1000)
        sensor_columns = ['Rv', 'Yv', 'Bv', 'Ri', 'Yi', 'Bi', 'X', 'Y', 'Z']
        self.feature_scaler.fit(df_sample[sensor_columns].astype(np.float32))
        rul_values = np.arange(len(df_sample), 0, -1)
        self.rul_scaler.fit(rul_values.reshape(-1, 1))

        while True:
            for file_path in file_paths:
                label = self.class_names.index(file_path.split('_')[-1].split('.')[0])
                chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
                
                for chunk in chunk_iterator:
                    sequences = []
                    labels_rul = []
                    labels_class = []

                    if not all(col in chunk.columns for col in sensor_columns):
                        continue
                    
                    sensor_data = chunk[sensor_columns].astype(np.float32)
                    scaled_data = self.feature_scaler.transform(sensor_data)

                    for i in range(len(scaled_data) - self.sequence_length - self.prediction_horizon + 1):
                        sequences.append(scaled_data[i:i + self.sequence_length])
                        rul_idx = i + self.sequence_length + self.prediction_horizon - 1
                        labels_rul.append(self.rul_scaler.transform([[len(scaled_data) - rul_idx]])[0])
                        labels_class.append(label)
                    
                        if len(sequences) >= batch_size:
                            X_batch = np.array(sequences[:batch_size])
                            y_rul_batch = np.array(labels_rul[:batch_size])
                            y_class_batch = to_categorical(labels_class[:batch_size], self.num_classes)
                            yield X_batch, {'classification': y_class_batch, 'rul': y_rul_batch}
                            sequences, labels_rul, labels_class = [], [], []
