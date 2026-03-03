#!/bin/bash
KERNEL="softkleenex/aimo-pp3-submission-baseline"
echo "Monitoring status for $KERNEL..."

while true; do
    STATUS=$(/Library/Frameworks/Python.framework/Versions/3.12/bin/kaggle kernels status $KERNEL | grep -o 'KernelWorkerStatus\.[A-Z]*')
    echo "[$(date +%T)] Current Status: $STATUS"
    
    if [[ "$STATUS" == "KernelWorkerStatus.COMPLETE" ]]; then
        echo "✅ Kernel execution finished successfully!"
        exit 0
    elif [[ "$STATUS" == "KernelWorkerStatus.ERROR" ]]; then
        echo "❌ Kernel execution failed with an error."
        exit 1
    elif [[ -z "$STATUS" ]]; then
        echo "⚠️ Could not retrieve status. Retrying..."
    fi
    
    sleep 60
done
