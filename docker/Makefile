python_version=3.8
registry=alezana
image_name=dg_base
dockerfile=Dockerfile.base

tag=$(registry)/$(image_name):$(python_version)

build:
	docker build -f $(dockerfile) -t $(tag) . --build-arg PYTHON_VERSION_=${python_version}

build-no-cache:
	docker build --no-cache -f $(dockerfile) -t $(tag) . --build-arg PYTHON_VERSION_=${python_version}

push-on-dockerhub:
	docker login
	docker push $(tag)