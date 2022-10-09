#!/bin/bash

OUTPUT=$(yc iam create-token)
echo "${OUTPUT}"