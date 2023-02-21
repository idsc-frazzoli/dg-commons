DOCKER_REGISTRY ?= docker.io
tagname="3.11"
python_version="3.11.2"
user=idscfrazz
image_name=dg_base
dockerfile=Dockerfile.base

tag=$(DOCKER_REGISTRY)/$(user)/$(image_name):$(tagname)

build:

	docker buildx build  --platform linux/arm64,linux/amd64 -f $(dockerfile) -t $(tag) --build-arg PYTHON_VERSION_=${python_version} --progress plain --push .

build-no-cache:
	docker buildx build  --platform linux/arm64,linux/amd64  -f $(dockerfile) -t $(tag)  --build-arg PYTHON_VERSION_=${python_version} --push . --no-cache

push-on-dockerhub:
	docker login
	docker push $(tag)
