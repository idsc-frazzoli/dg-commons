DOCKER_REGISTRY ?= docker.io
tagname="3.11"
python_version="3.11"
user=idscfrazz
image_name=dg_base
dockerfile=Dockerfile.base

tag=$(DOCKER_REGISTRY)/$(user)/$(image_name):$(tagname)
# platforms=linux/arm64,linux/amd64
platforms=linux/amd64

build:
	docker buildx build  --secret ID=PIP_INDEX_URL --platform $(platforms) -f $(dockerfile) -t $(tag) --build-arg PYTHON_VERSION_=${python_version}  --push .

build-no-cache:
	docker buildx build  --secret ID=PIP_INDEX_URL  --platform $(platforms)  -f $(dockerfile) -t $(tag)  --build-arg PYTHON_VERSION_=${python_version} --push . --no-cache

push-on-dockerhub:
	docker login
	docker push $(tag)
