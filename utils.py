import glob
#import unidecode
import unicodedata
import random
import time
import math
import sys

# get files from the given path
def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip().lower()) for line in some_file]

# get the number of lines from a text file
def getLines(f):
    lines = readLines(f)
    print('lines: ', len(lines), ' -> ', f)
    return lines

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# return difference between current time and given time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# convert seconds to minutes
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# return difference between current time and given time
def timeSinceStart(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

# Print function for showing progress as percentage
def progressPercent(totalNames, start, names, p, samplesGenerated):
    bar_len = 50
    filled_len = int(round(bar_len * names / float(totalNames)))
    percents = round(100.0 * names / float(totalNames), 1)
    nNames = int(p / 100 * totalNames)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write(
        '[%s] %s%s names founded among %d samples generated (%d of %d names) on %s (goal = %.1f%% = %d names)\r' % (
        bar, percents, '%', samplesGenerated, names, totalNames, timeSinceStart(start), p, nNames))
    sys.stdout.flush()

# Print function for showing progress as percentage 
def progress(total, acc, start, epoch, l):
    bar_len = 50
    filled_len = int(round(bar_len * epoch / float(total)))
    percents = round(100.0 * epoch / float(total), 1)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s epoch: %d acc: %.3f %% and testing size = %d names => coverage of %.3f %% on %s \r' % (
    bar, percents, '%', epoch, (100 * acc / epoch), l, (100 * acc / l), timeSinceStart(start)))
    sys.stdout.flush()

# return the mean of the number of character per line
def getMeanSize(listData):
    mean = 0
    for word in listData:
        mean = mean + len(word)

    return int(mean / len(listData))