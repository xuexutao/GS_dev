#!/bin/bash

msg=${1:-"Regular update"}

git add .
git commit -m "$msg"
git push


# git push --set-upstream origin xutao
