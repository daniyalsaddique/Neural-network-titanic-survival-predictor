import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data

def train_model():
    # Load data
    X_train_scaled, y_train, _ = preprocess_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
    
    # Save model
    model.save('titanic_model.h5')

if _name_ == "_main_":
    train_model()
