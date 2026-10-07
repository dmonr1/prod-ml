from app.services.training_service import train_models


def main() -> None:
    train_models(("course",))


if __name__ == "__main__":
    main()
