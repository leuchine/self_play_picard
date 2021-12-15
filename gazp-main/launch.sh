#!/bin/bash

docker run -it --rm \
  -m8g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test bash
