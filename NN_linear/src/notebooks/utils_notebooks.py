import os
has_path_been_modified: bool = False


def move_current_path_up(n_times: int = 1):
    """ Moves the current working directory up into the folders. Mostly useful for notebooks

    :param n_times: Number of folders to crawl up
    """
    global has_path_been_modified

    if not has_path_been_modified:
        project_path = os.getcwd()

        for _ in range(n_times):
            project_path = os.path.dirname(project_path)

        os.chdir(project_path)

    has_path_been_modified = True
    print('New working directory = %s' % os.getcwd())
