    def train(self, file_paths, batch_size=None, epochs=10):
        """Train the model with memory efficiency."""
        batch_size = batch_size or self.batch_size
        train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)
        
        self.model = self.build_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            self.data_generator(train_files, batch_size=batch_size),
            steps_per_epoch=len(train_files),
            validation_data=self.data_generator(val_files, batch_size=batch_size),
            validation_steps=len(val_files),
            epochs=epochs,
            callbacks=callbacks
        )
        return history
