"""Train and evaluate both models from the canonical synthetic CSV files."""
from app.services.training_service import train_models


def main() -> None:
    result = train_models()
    print(f"Modelos sintéticos publicados: {result['run_id']}")


if __name__ == "__main__":
    main()
