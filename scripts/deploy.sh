#!/bin/bash

# Deploy script to update the Kaggle dataset with latest source code
# Usage: ./scripts/deploy.sh "Commit message"

MSG=${1:-"Update source code"}

echo "Deploying src/ to Kaggle Dataset..."
kaggle datasets version -p src -m "$MSG" --dir-mode zip

if [ $? -eq 0 ]; then
    echo "Deployment successfully started!"
    echo "Check status at: https://www.kaggle.com/datasets/softkleenex/aimo-pp3-source"
else
    echo "Deployment failed."
fi
