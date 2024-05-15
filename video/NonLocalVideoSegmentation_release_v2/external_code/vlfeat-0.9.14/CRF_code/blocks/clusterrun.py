#!/usr/bin/python

from optparse import OptionParser
import datetime
import time
import subprocess
import os
import sys
import math

parser = OptionParser()

parser.add_option(
    "-s", "--script", 
    dest    = "scriptlet",
    default = "fprintf('Empty scriptlet (%d).\n', cluster_job_id);",
    help    = "specify scriptlet to use", 
    metavar = "FILE")

parser.add_option(
    "-l", "--log", 
    dest    = "log_prefix", 
    default = None,
    help    = "specify the log files prefix",
    metavar = "FILE")

parser.add_option(
    "-j", "--jobs", 
    dest    = "num_jobs",
    default = 1,
    type    = "int",
    help    = "specify the number of jobs to dispatch",
    metavar = "NUM")

parser.add_option(
    "-d", "--debug", 
    dest    = "debug",
    default = False,
    action  = "store_true",
    help    = "print debug informations")

# --------------------------------------------------------------------
class Host:
# --------------------------------------------------------------------

    def __init__ (self, url):
        self.url     = url
        self.job     = None
        self.process = None

    def start(self, job):
        self.job = job
        
        bs = ""
        bs = bs + "TERM=vt100 ssh '%s' -t -t <<EOF\n" % self.url
        bs = bs + self.job.script
        bs = bs + "EOF\n"
        bs = bs + "exit $?\n"
   
        logfile = self.job.getLogFile()
        if self.job.getLogFile():
            logfile.write("==== BEGIN SHELL-SCRIPT ====\n") ;
            logfile.write(bs)
            logfile.write("==== END SHELL-SCRIPT ====\n") ;
            logfile.flush()

        self.process = subprocess.Popen(
            bs,
            cwd       = os.path.abspath(os.curdir),
            env       = os.environ,
            stdout    = logfile,
            stderr    = logfile,
            close_fds = False,
            shell     = True)

        self.job.host    = self
        self.job.status  = Job.RUN
        self.job.started = time.time()
        
        print "START: %s" % self.job

    def poll(self):
        if not self.job:
            return None        
        ret = self.process.poll()
        if self.job.status == Job.RUN:
            if ret != None:
                self.job.ended = time.time()
                if ret == 0:
                    self.job.status = Job.DONE
                else:
                    self.job.status = Job.FAIL
                print "END:   %s" % self.job
        return self.job.status

    def __str__ (self):
        msg  = "%10s" % self.url
        if not self.job:
            msg = msg + " ---"
        else:
            msg = msg +  "%5d %5s" % (self.job.number, 
                                      jobStatusToString(self.job.status))
        return msg

# --------------------------------------------------------------------
def findFreeHost():
# --------------------------------------------------------------------
    while True:
        for h in hosts:
            if h.poll() != Job.RUN:
                return h
        time.sleep(5)

# --------------------------------------------------------------------
def waitAllHosts():
# --------------------------------------------------------------------
    waitMore = True
    while waitMore:
        waitMore = False
        for h in hosts:
            if h.poll() == Job.RUN:
                waitMore = True
        time.sleep(5)

# --------------------------------------------------------------------
def jobStatusToString(s):
# --------------------------------------------------------------------
    if s == Job.WAIT:
        return "WAIT"
    elif s == Job.RUN:
        return "RUN"
    elif s == Job.FAIL:
        return "FAIL"
    elif s == Job.DONE:
        return "DONE"
    else:
        return "????"

# --------------------------------------------------------------------
class Job:
# --------------------------------------------------------------------
    WAIT = 1
    RUN  = 2
    FAIL = 3
    DONE = 4
    
    def __init__ (self,
                  number,
                  scriptlet,
                  logname = None):
        
        self.host      = None
        self.started   = None
        self.ended     = None    
        self.status    = Job.WAIT

        self.number    = number
        self.scriptlet = scriptlet
        self.logname   = logname
        self.logfile   = None

        self.script = """
export PS1="<-> "
export PS2="<-> "
cd '%s' ;
mkdir -p /tmp/bk-\${USER} ;
TAG_FILE=/tmp/bk-\${USER}/\$\$.tag
touch \${TAG_FILE}
echo "** TAG_FILE = \${TAG_FILE}" ;
make ;
matlab -nojvm -nodisplay <<DONE
try
  cluster_job_id=%d ;
  blocks_setup ;
  %s
  unix('rm -f "\${TAG_FILE}"') ;
catch
  e=lasterror;
  s=e.stack ;
  fprintf('\\nIn:\\n') ;
  for q=1:length(s)
    fprintf('%%s:%%d @ %%s\\n', s(q).file, s(q).line, s(q).name) ;
  end
  fprintf('ERROR: (%%s) %%s\\n', e.identifier, e.message) ;
  unix(sprintf('kill %%d', vl_getpid)) ;
end
exit
DONE
succ=\$?
echo "** MATLAB return code: \$succ"
if test -e \${TAG_FILE} ;
then
  echo "** MATLAB did not remove tag file \${TAG_FILE}" ;
  succ=1 ;
fi
exit \$succ
""" % (os.path.abspath(os.curdir), self.number, self.scriptlet)
        
    def getLogFile(self):
        if not self.logname:
            return None        
        if not self.logfile:
            self.logfile = open(self.logname, "w")
            if not self.logfile:
                print "Warning: could not open log file '%s'" % self.logname

        return self.logfile
            

    def __str__ (self):
        msg = "%5d %5s" % (self.number, jobStatusToString(self.status))
        if self.host:
            msg = msg + "%10s" % self.host.url
        else:
            msg = msg + "%10s" % "..."
        if self.started and self.ended:
            dur = self.ended - self.started
            min = math.floor(dur / 60)
            sec = dur - min * 60
            str = "%02d:%02d" % (min, sec)
            msg = msg + "%7s" % str
        else:
            msg = msg + "%7s" % "..."
        return msg

# --------------------------------------------------------------------
if __name__ == "__main__":
# --------------------------------------------------------------------

    hosts = [
        Host("localhost"),
        Host("localhost"),
    ]

    jobs = []

    (options, args) = parser.parse_args()

    def printDebug():
        if options.debug:
            print "Hosts:"
            for h in hosts:
                print h
            print "Jobs:"
            for j in jobs:
                print j

    num_jobs = options.num_jobs

    for j in range(1,num_jobs+1):
        jobs.append(Job(j, 
                        options.scriptlet,
                        "%s.%04d" % (options.log_prefix,j)))

    next_job = 1    
    while next_job <= num_jobs:
        printDebug()
        h = findFreeHost()     
        h.start(jobs[next_job - 1])
        next_job = next_job + 1
    
    waitAllHosts()
    printDebug()

    for j in jobs:
        if j.status == Job.FAIL:
            print "Error:"
            print j
            print "job terminated with a failure state."
            sys.exit(1)            
    sys.exit(0)
