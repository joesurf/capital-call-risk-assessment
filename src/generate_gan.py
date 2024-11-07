import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


df_realistic = pd.read_csv('synthetic_data.csv')

def calculate_age(dob):
    birth_year = int(dob.split('.')[2])
    current_year = 2024  
    age = current_year - birth_year
    return age

def preprocess_data(df):
    df['Age'] = df['Date of birth'].apply(calculate_age)
    df = df.drop(columns=['Id', 'Date of birth'])

    df = pd.get_dummies(df, columns=["Source of wealth other", "Source of funds other"], drop_first=True)

    df = df.replace({True: 1, False: 0})

    return df

def scale_data(df):
    df = (df - df.min()) / (df.max() - df.min())

    return df

df_realistic = preprocess_data(df_realistic)
df_realistic_processed = scale_data(df_realistic)

def create_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=input_dim, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    return model

def create_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(1024, input_dim=input_dim, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

latent_dim = 128
data_dim = df_realistic_processed.shape[1]

generator = create_generator(latent_dim, data_dim)
discriminator = create_discriminator(data_dim)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

train_gan(generator, discriminator, gan, df_realistic_processed.values)

noise = np.random.normal(0, 1, (100, latent_dim))
generated_data = generator.predict(noise)

new_data = (generated_data * (df_realistic.max().values - df_realistic.min().values)) + df_realistic.min().values
new_data = pd.DataFrame(new_data, columns=df_realistic.columns)


new_data[new_data.columns.difference(['Age'])] = new_data[new_data.columns.difference(['Age'])] >= 0.5

new_data.to_csv('gan_data.csv', index=False)