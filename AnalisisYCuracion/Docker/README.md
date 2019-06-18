## Contenido:

* Dockerfile
* Requerimientos de python
* Carpeta con datasets
* Carpeta con notebooks 

### Comandos DOCKER:

* __BUILD:__
```sh
$ docker build -t retail_docker .
```

* __RUN:__

```sh
$ docker run -it -p 8888:8888 retail_docker
```

Luego de correr el contenedor, se debe ingresar a la url indicada en consola
