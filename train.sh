# Fetch dataset
cd dataset
bash fetch_imagenet.sh
cd ..

# Train the model
cd context_cluster
python -m train --config /notebooks/args.yaml
