Installing RVM to /usr/local/rvm/
Installation of RVM in /usr/local/rvm/ is almost complete:

  * First you need to add all users that will be using rvm to 'rvm' group,
    and logout - login again, anyone using rvm will be operating with `umask u=rwx,g=rwx,o=rx`.

  * To start using RVM you need to run `source /etc/profile.d/rvm.sh`
    in all your open shell windows, in rare cases you need to reopen all shell windows.
120:export PATH=$PATH:/home/abhineet/bin:/home/abhineet/.local/bin:/home/abhineet/bin:/home/abhineet/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/abhineet/scripts:/home/abhineet/pycharm-2017.2.3/bin:/usr/local/cuda-8.0/bin/

  * WARNING: Your '/home/abhineet/.bashrc' contains `PATH=` with no `$PATH` inside, this can break RVM,
    for details check https://github.com/rvm/rvm/issues/1351#issuecomment-10939525
    to avoid this warning prepend `$PATH`.

root@abhineet-VirtualBox:~/H/UofA/617/Project/code# source ~/.rvm/scripts/rvm
bash: /home/abhineet/.rvm/scripts/rvm: No such file or directory
root@abhineet-VirtualBox:~/H/UofA/617/Project/code# source /etc/profile.d/rvm.sh
root@abhineet-VirtualBox:~/H/UofA/617/Project/code# 

