/* Experiment helper (E3b): switch to an unprivileged uid and re-apply an environment captured by the
 * traced parent, so Nsight Systems' injection variables survive the privilege change (glibc strips
 * LD_PRELOAD from a file-capability binary). Only the spawner's uid may call it; the target uid must be
 * above it; an optional :GID keeps a shared group (E3c). Capabilities are cleared before exec.
 * Usage: dropuid UID[:GID] ENVFILE PROGRAM [ARGS...] */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/capability.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SPAWNER_UID 1000

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: dropuid UID ENVFILE PROGRAM [ARGS...]\n"); return 2; }
    if (getuid() != SPAWNER_UID) { fprintf(stderr, "dropuid: caller is not the spawner\n"); return 3; }
    char *colon = strchr(argv[1], ':');
    uid_t target = (uid_t)strtoul(argv[1], NULL, 10);
    gid_t group = colon ? (gid_t)strtoul(colon + 1, NULL, 10) : (gid_t)target;
    if (target <= SPAWNER_UID || target > 65000) { fprintf(stderr, "dropuid: refused uid %u\n", target); return 3; }
    if (group < SPAWNER_UID || group > 65000) { fprintf(stderr, "dropuid: refused gid %u\n", group); return 3; }
    /* read the captured environment: NUL- or newline-separated KEY=VALUE lines */
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("dropuid: envfile"); return 4; }
    struct stat st; fstat(fd, &st);
    char *buf = malloc(st.st_size + 1);
    ssize_t n = read(fd, buf, st.st_size); close(fd);
    if (n < 0) { perror("dropuid: read"); return 4; }
    buf[n] = 0;
    char **env = calloc(n + 2, sizeof(char *)); int count = 0;
    for (char *line = strtok(buf, "\n"); line; line = strtok(NULL, "\n"))
        if (strchr(line, '=')) env[count++] = line;
    env[count] = NULL;
    if (setresgid(group, group, group) != 0 || setgroups(0, NULL) != 0 || setresuid(target, target, target) != 0) {
        perror("dropuid: setres[ug]id"); return 5;
    }
    struct __user_cap_header_struct hdr = { _LINUX_CAPABILITY_VERSION_3, 0 };
    struct __user_cap_data_struct data[2] = { {0, 0, 0, 0}, {0, 0, 0, 0} };
    if (syscall(SYS_capset, &hdr, data) != 0) { perror("dropuid: capset"); return 5; }
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    if (getuid() != target || geteuid() != target) { fprintf(stderr, "dropuid: uid not switched\n"); return 5; }
    execve(argv[3], argv + 3, env);
    perror("dropuid: execve"); return 6;
}
