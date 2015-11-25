from . import ais_code as ais
import mex

class Annotation(object):
    """docstring for Annotation"""
    def __init__(self, start, finish=None, name=None, info=None, punctuation=None, orbit=None):
        super(Annotation, self).__init__()
        self.start = start
        if finish is None:
            self.finish = self.start
        else:
            self.finish = finish
        self.name = name
        self.info = info
        self.punctuation = punctuation

def read_annotations(f=None):
    """docstring for read_anotation"""
    if f is None:
        f = mex.data_directory + 'ais_annotations.txt'

    definitions = dict()
    punctuation = "+-?*"

    begin_data = False
    events = []
    numbers = '0 1 2 3 4 5 6 7 8 9'.split(' ')

    with open(f, 'r') as f:
        for line in f:
            if (not line) or (line == '\n'):
                continue

            if line[0] == '#':
                continue


            if not begin_data:
                if 'BEGIN_OBSERVATIONS' in line:
                    begin_data = True
                    continue

                try:
                    k, v = line.split(":")
                    definitions[k.lstrip().rstrip()] = v.lstrip().rstrip()
                except ValueError as e:
                    pass

            else:
                t, s = line.split(" ", 1)

                s = s.lstrip().rstrip()
                t = t.lstrip().rstrip()

                if "." in t:
                    time = float(t)
                elif "T" in t:
                    time = celsius.utcstr_to_spiceet(t)
                else:
                    for c in t:
                        if not c in numbers:
                            raise ValueError("Couldn't understand time: %s" % t)
                    time = mex.orbits[int(t)].start + 100. # 100 seconds to avoid float near-equality

                tags = []
                if s.endswith('"'):
                    info = s.lstrip('"')
                    s = s.split('"')[0]
                else:
                    info = None

                for w in s.split(', '):
                    if not w:
                        continue
                    wp = w.rstrip(punctuation)
                    if not wp in list(definitions.keys()):
                        raise KeyError("%s not a recognised definition" % wp)

                    if w[-1] in punctuation:
                        p = w[-1]
                    else:
                        p = None

                    tags.append([wp, p])

                orbit = mex.orbits[time]
                print(line, orbit.number)

                for t in tags:
                    e = Annotation(time, orbit.finish, name=t[0], info=info, punctuation=p, orbit=orbit.number)
                    events.append(e)

    return events, definitions


