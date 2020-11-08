mkdir -p ~/.streamlit/
cat ~/.streamlit/config.toml

streamlit run \
    app.py \
    --server.port $PORT \
    --server.enableCORS false \
    --server.headless true
