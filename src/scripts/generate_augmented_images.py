from src.utils.hands_augmentation import HandsAugmentation

ROOT_DIR = 'dataset'
LABELS = {0: 'paper', 1: 'rock', 2: 'scissors'}
MULTIPLY_SCALER = 5

def main():

    hands_augmentation = HandsAugmentation(
        root_dir=ROOT_DIR,
        labels=LABELS,
        multiply_scaler=MULTIPLY_SCALER
    )

    hands_augmentation.gen_augmented_images()


if __name__ == '__main__':
    main()