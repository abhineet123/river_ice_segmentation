#!/bin/bash
git.exe --git-dir=.gitb add --all .
if [ "$#" -ne 1 ]; then
   git.exe --git-dir=.gitb commit
else
	if [ "$1" == "f" ]; then
		git.exe --git-dir=.gitb commit -m "minor fix"
	elif [ "$1" == "r" ]; then
		git.exe --git-dir=.gitb commit -m "Updated README"
	else
		git.exe --git-dir=.gitb commit -m "$1"
	fi
fi
git.exe --git-dir=.gitb push origin master --force

