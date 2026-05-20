import os
import logging
import mlflow


def main():
    logging.basicConfig(level=logging.INFO)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    logging.info("Using MLFLOW_TRACKING_URI=%s", uri)

    with mlflow.start_run() as run:
        mlflow.log_param("example_param", "hello")
        for i in range(3):
            mlflow.log_metric("example_metric", i * 0.5, step=i)

        artifact_path = "artifact.txt"
        with open(artifact_path, "w") as f:
            f.write("This is a test artifact for MLflow.\n")

        mlflow.log_artifact(artifact_path)
        logging.info("Logged run: %s", run.info.run_id)


if __name__ == "__main__":
    main()