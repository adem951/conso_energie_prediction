param(
    [int]$Port = 5000
)

mlflow server `
    --backend-store-uri "sqlite:///mlflow.db" `
    --host 127.0.0.1 `
    --port $Port