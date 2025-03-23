#!/bin/bash

kubectl delete deployments --all
kubectl delete pods --all

docker build -t dabblu/edge_image ./edge
docker push dabblu/edge_image
docker build -t dabblu/fog_image ./fog
docker push dabblu/fog_image
docker build -t dabblu/cloud_image ./cloud
docker push dabblu/cloud_image
docker build -t dabblu/fog_image_2 ./fog_node_2
docker push dabblu/fog_image_2
docker build -t dabblu/fog_image_3 ./fog_node_3
docker push dabblu/fog_image_3


kubectl apply -f edge/edge_deployment.yaml
kubectl apply -f fog/fog_deployment.yaml
kubectl apply -f cloud/cloud_deployment.yaml
kubectl apply -f fog_node_2/fog_deployment.yaml
kubectl apply -f fog_node_3/fog_deployment.yaml



