Set:

```
VERSION=1.0.0
REPO_URL=paolamedo/dash
BUILD_DIR=/home/paola/Documents/MCD/MNO/tsp-dash
```
Use:
```
docker pull $REPO_URL:$VERSION
```

Build:

```
docker build $(pwd) -f Dockerfile-dash --force-rm -t $REPO_URL:$VERSION
```

Upload to Dockerhub:
```
docker login
docker push $REPO_URL:$VERSION
```


Run:

```
docker run --rm -it \
--name dashy \
-p 8050:8050 \
-v $BUILD_DIR:/home \
 $REPO_URL:$VERSION
```
