class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def schedule_verbose(POLICY):
    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print(bcolors.FAIL + POLICY['says'] + bcolors.ENDC)
    for key in POLICY:
        if key in ['BATCH_SIZE',
                   'step_values',
                   'learning_rates',
                   'momentum',
                   'decay',
                   'max_iter',
                   'object_scale',
                   'noobject_scale',
                   'class_scale',
                   'coord_scale',
                   'thresh ',
                   'num',
                   'anchors']:
            print(bcolors.BOLD + str(key) + bcolors.ENDC)
            print(str(POLICY[str(key)]).rjust(80))
    print("\n" * 2)

    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print(bcolors.WARNING + 'network' + bcolors.ENDC)
    for key in POLICY:
        if key in ['height',
                   'width',
                   'channels',
                   'side',
                   'interpolation']:
            print(bcolors.BOLD + str(key) + bcolors.ENDC)
            print(str(POLICY[str(key)]).rjust(80))
    print(bcolors.WARNING + "#" * 80 + bcolors.ENDC)
    print("\n" * 2)

