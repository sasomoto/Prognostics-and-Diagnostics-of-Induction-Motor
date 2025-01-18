    def build_model(self):
        """Build a hybrid CNN-LSTM model."""
        input_layer = Input(shape=(self.sequence_length, 9))
        
        # CNN Layers
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # LSTM Layer
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        # Classification Output
        class_output = Dense(32, activation='relu')(x)
        class_output = Dropout(0.3)(class_output)
        class_output = Dense(self.num_classes, activation='softmax', name='classification')(class_output)
        
        # RUL Prediction Output
        rul_output = Dense(32, activation='relu')(x)
        rul_output = Dropout(0.3)(rul_output)
        rul_output = Dense(1, activation='sigmoid', name='rul')(rul_output)
        
        model = Model(inputs=input_layer, outputs=[class_output, rul_output])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'classification': 'categorical_crossentropy', 'rul': 'mse'},
            metrics={'classification': ['accuracy'], 'rul': ['mae']}
        )
        return model
