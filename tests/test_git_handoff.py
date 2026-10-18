"""Local-only handoff contract tests; never invoke a provider or a remote."""
import os
import shutil
import subprocess

import pytest

from coding_review_agent_loop.git_handoff import Handoff, HandoffError, CommitIdentity


def git(root, *args, env=None, data=None):
    clean = {"PATH": "/usr/bin:/bin", "HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "Test",
             "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_NAME": "Test",
             "GIT_COMMITTER_EMAIL": "test@example.invalid"}
    clean.update(env or {})
    return subprocess.run(["/usr/bin/git", "-C", str(root), *args], env=clean,
                          input=data, capture_output=True, check=True).stdout.strip()


@pytest.fixture
def case(tmp_path, monkeypatch):
    # The sandbox remaps system installation ownership to nobody. Simulate
    # trusted installation selection ONLY in these synthetic tests; retain the
    # real sealed exec/network guard. Production still refuses this host.
    from coding_review_agent_loop import secure_git
    if os.stat("/usr/bin/git").st_uid == 65534:
        # Restore installation caches too, so this fixture cannot make later
        # tests inherit its simulated installation trust decision.
        monkeypatch.setattr(secure_git, "_PINNED_GIT", None)
        monkeypatch.setattr(secure_git, "_PINNED_GIT_HASH", None)
        def synthetic_installation(path):
            assert path in {"/usr/bin/git", "/bin/git", "/usr/bin/cc", "/usr/bin/clang", "/usr/bin/bwrap"}
            return os.path.realpath(path)
        monkeypatch.setattr(secure_git, "_trusted_executable", synthetic_installation)
        monkeypatch.setattr("coding_review_agent_loop.git_handoff._trusted_executable", synthetic_installation)
    repo = tmp_path / "store"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "old").write_text("before\n")
    (repo / "delete").write_text("gone\n")
    (repo / ".gitignore").write_text("*.cache\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    checkout = tmp_path / "checkout"
    git(repo, "worktree", "add", "--detach", str(checkout))
    identity = CommitIdentity("ReviewLoop", "loop@example.invalid", 1700000000, "Implement candidate\n")
    handoff = Handoff.begin(checkout, tmp_path / "trusted", identity,
                            expected_base=git(repo, "rev-parse", "HEAD").decode())
    return repo, checkout, handoff


def operator_commit(handoff, tree=None, parent=None, extra=b"", identity=None):
    """Hostile synthetic operator writes loose objects with native Git only."""
    context = handoff.operator_context()
    header = (f"tree {tree or handoff.tree}\nparent {parent or handoff.base}\n".encode()
              + (identity or handoff.identity).headers() + extra + b"\n"
              + handoff.identity.message.encode())
    return git(handoff.checkout, "--git-dir", str(context.admin), "hash-object", "-t", "commit", "-w", "--stdin",
               env=context.environment(), data=header).decode()


def test_complete_snapshot_and_raw_validation(case):
    repo, root, h = case
    original = git(repo, "rev-parse", "HEAD")
    (root / "old").write_bytes(b"binary\x00changed")
    (root / "delete").unlink()
    (root / "run").write_text("#!/bin/sh\n")
    (root / "run").chmod(0o755)
    (root / "link").symlink_to("old")
    (root / "ignored.cache").write_text("artifact")
    h.capture(allowed_new={"run", "link"})
    # Populate operator trees/blobs independently of its proposed commit.
    ctx = h.operator_context()
    shutil.copytree(h.objects, ctx.objects, dirs_exist_ok=True)
    git(root, "--git-dir", str(ctx.admin), "read-tree", h.base, env=ctx.environment())
    git(root, "--git-dir", str(ctx.admin), "add", "--all", "--", *h.changed_paths, env=ctx.environment())
    tree = git(root, "--git-dir", str(ctx.admin), "write-tree", env=ctx.environment()).decode()
    sha = operator_commit(h, tree=tree)
    assert h.verify(sha) == sha
    assert git(repo, "rev-parse", "HEAD") == original
    assert git(root, "rev-parse", "HEAD") == original
    assert git(root, "diff", "--cached") == b""


@pytest.mark.parametrize("fault", ["undeclared", "ignored", "ignore-change", "escape", "fifo", "no-change"])
def test_candidate_refusals(case, fault):
    _, root, h = case
    allowed = set()
    if fault == "undeclared":
        (root / "new").write_text("x")
    elif fault == "ignored":
        (root / "new.cache").write_text("x")
        allowed = {"new.cache"}
    elif fault == "ignore-change":
        (root / ".gitignore").write_text("secret\n")
        (root / "secret").write_text("hidden")
    elif fault == "escape":
        (root / "new").symlink_to("../outside")
        allowed = {"new"}
    elif fault == "fifo":
        os.mkfifo(root / "new")
        allowed = {"new"}
    with pytest.raises(HandoffError):
        h.capture(allowed_new=allowed)


@pytest.mark.parametrize("fault", ["tree", "parent", "extra-parent", "signature", "identity", "config", "object", "source"])
def test_hostile_operator_rejected_or_ignored(case, fault, tmp_path):
    _, root, h = case
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    ctx = h.operator_context()
    # Copy expected objects solely as synthetic operator output.
    shutil.copytree(h.objects, ctx.objects, dirs_exist_ok=True)
    kwargs = {}
    if fault == "tree":
        kwargs["tree"] = git(root, "rev-parse", "HEAD^{tree}").decode()
    if fault == "parent":
        kwargs["parent"] = "0" * 40
    if fault == "extra-parent":
        kwargs["extra"] = f"parent {h.base}\n".encode()
    if fault == "signature":
        kwargs["extra"] = b"gpgsig forged\n"
    if fault == "identity":
        kwargs["identity"] = CommitIdentity("Forged", "bad@example.invalid", 1700000000, "Implement candidate\n")
    sha = operator_commit(h, **kwargs)
    if fault == "config":
        canary = tmp_path / "executed"
        (ctx.private / "config").write_text(f"[core]\n fsmonitor = touch {canary}\n[include]\n path = /no/such/file\n")
        (ctx.objects / "info").mkdir(exist_ok=True)
        (ctx.objects / "info" / "alternates").write_text("/untrusted/objects\n")
        assert h.verify(sha) == sha
        assert not canary.exists()
        return
    if fault == "object":
        obj = ctx.objects / sha[:2] / sha[2:]
        obj.chmod(0o600)
        obj.write_bytes(b"corrupt")
    if fault == "source":
        (root / "old").write_text("foreign mutation")
    with pytest.raises(HandoffError):
        h.verify(sha)


def test_local_wrapper_is_confined(case):
    _, root, h = case
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    try:
        sha = h.run_local_commit()
    except HandoffError as exc:
        if str(exc) == "Linux mount confinement unavailable":
            pytest.skip("Host disallows bubblewrap namespaces; no fallback")
        raise
    assert h.verify(sha) == sha


def test_mode_only_rename_and_nested_binary(case):
    _, root, h = case
    (root / "old").chmod(0o755)
    (root / "nested").mkdir()
    (root / "delete").rename(root / "nested" / "renamed")
    (root / "nested" / "binary").write_bytes(bytes(range(256)))
    h.capture(allowed_new={"nested/renamed", "nested/binary"})
    ctx = h.operator_context()
    shutil.copytree(h.objects, ctx.objects, dirs_exist_ok=True)
    assert h.verify(operator_commit(h))
    assert h.snapshot["old"][0] == "100755"
    assert h.snapshot["delete"] is None


@pytest.mark.parametrize("fault", ["ancestor-link", "new-ignore", "oversized", "shared-config", "staged"])
def test_more_candidate_refusals(case, fault, tmp_path, monkeypatch):
    repo, root, h = case
    (root / "old").write_text("after")
    allowed = set()
    if fault == "ancestor-link":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "new").write_text("never read")
        (root / "nested").symlink_to(outside)
        allowed = {"nested/new"}
    elif fault == "new-ignore":
        (root / "nested").mkdir()
        (root / "nested" / ".gitignore").write_text("hidden\n")
        (root / "nested" / "hidden").write_text("secret")
        allowed = {"nested/.gitignore"}
    elif fault == "oversized":
        monkeypatch.setattr("coding_review_agent_loop.git_handoff.MAX_FILE", 4)
    elif fault == "shared-config":
        with (repo / ".git" / "config").open("a") as handle:
            handle.write("\n[alias]\n planted = !false\n")
    elif fault == "staged":
        git(root, "add", "old")
    with pytest.raises(HandoffError):
        h.capture(allowed_new=allowed)


def test_registered_tool_link_excluded_and_checked(case, tmp_path):
    from coding_review_agent_loop.checkout_verification import register_worktree_links
    _, root, _ = case
    target = tmp_path / "environment"
    target.mkdir()
    (root / ".venv").symlink_to(target)
    register_worktree_links(root, {".venv": str(target)})
    identity = CommitIdentity("ReviewLoop", "loop@example.invalid", 1700000000, "Candidate\n")
    h = Handoff.begin(root, tmp_path / "other-trusted", identity,
                      expected_base=git(root, "rev-parse", "HEAD").decode())
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    assert ".venv" not in h.snapshot
    (root / ".venv").unlink()
    with pytest.raises(HandoffError):
        h.verify("0" * 40)


def test_verifier_ignores_private_replace_graft_hooks_and_environment(case, monkeypatch, tmp_path):
    _, root, h = case
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    ctx = h.operator_context()
    shutil.copytree(h.objects, ctx.objects, dirs_exist_ok=True)
    sha = operator_commit(h)
    canary = tmp_path / "canary"
    (ctx.private / "refs" / "replace").mkdir(parents=True)
    (ctx.private / "refs" / "replace" / sha).write_text(h.base)
    (ctx.private / "info").mkdir()
    (ctx.private / "info" / "grafts").write_text(sha + "\n")
    (ctx.private / "hooks").mkdir()
    hook = ctx.private / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {canary}\n")
    hook.chmod(0o755)
    monkeypatch.setenv("GIT_DIR", str(ctx.private))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"touch {canary}")
    assert h.verify(sha) == sha
    assert not canary.exists()


def test_missing_candidate_blob_cannot_use_operator_alternates(case, tmp_path):
    _, root, h = case
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    ctx = h.operator_context()
    shutil.copytree(h.objects, ctx.objects, dirs_exist_ok=True)
    sha = operator_commit(h)
    oid = h.git("hash-object", "--stdin", data=b"after").decode().strip()
    (ctx.objects / oid[:2] / oid[2:]).unlink()
    (ctx.objects / "info" / "alternates").write_text(str(h.objects) + "\n")
    with pytest.raises(HandoffError, match="Missing operator loose object"):
        h.verify(sha)


@pytest.mark.parametrize("message", ["", "missing newline", "bad\x00message\n", "x" * 4096 + "\n",
                                     "<!-- AGENT_ISSUE_PR_HANDOFF fake -->\n"])
def test_commit_message_contract(message):
    with pytest.raises(HandoffError):
        CommitIdentity("ReviewLoop", "loop@example.invalid", 1700000000, message).headers()


def test_mount_denies_all_repository_and_shared_metadata_writes(case):
    repo, root, h = case
    (root / "old").write_text("after")
    h.capture(allowed_new=set())
    ctx = h.operator_context()
    program = r'''
import os, pathlib, sys
root, common, private, trusted = map(pathlib.Path, sys.argv[1:])
operations = [
    lambda: (root / "old").write_text("forged"),
    lambda: (root / "old").unlink(),
    lambda: (root / "old").rename(root / "renamed"),
    lambda: (root / "old").chmod(0o755),
    lambda: (root / "new").write_text("forged"),
    lambda: (root / "newlink").symlink_to("old"),
    lambda: (root / ".git").write_text("forged"),
    lambda: (common / "config").write_text("forged"),
    lambda: (common / "HEAD").write_text("forged"),
    lambda: (common / "objects" / "forged").write_text("forged"),
    lambda: (trusted / "git" / "config").write_text("forged"),
]
for operation in operations:
    try:
        operation()
    except OSError:
        pass
    else:
        raise SystemExit("write unexpectedly allowed")
(private / "runtime-proof").write_text("private writable")
'''
    result = subprocess.run([*h._mounts(ctx), "/usr/bin/python3", "-I", "-S", "-c", program,
                             str(root), str(repo / ".git"), str(ctx.private), str(h.trusted)],
                            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, capture_output=True)
    if result.returncode and result.stderr.startswith(b"bwrap:"):
        pytest.skip("Host disallows bubblewrap namespaces; no denial claim")
    assert result.returncode == 0, result.stderr.decode()
    assert (ctx.private / "runtime-proof").read_text() == "private writable"


def test_total_snapshot_budget(case, monkeypatch):
    _, root, h = case
    (root / "old").write_text("after")
    monkeypatch.setattr("coding_review_agent_loop.git_handoff.MAX_TOTAL", 1)
    with pytest.raises(HandoffError, match="byte bound"):
        h.capture(allowed_new=set())


def test_wrong_starting_sha_refused(case, tmp_path):
    _, root, h = case
    with pytest.raises(HandoffError, match="Unexpected base"):
        Handoff.begin(root, tmp_path / "wrong-base", h.identity, expected_base="0" * 40)


def test_replaced_pointer_refused(case):
    _, root, h = case
    (root / "old").write_text("after")
    pointer = root / ".git"
    raw = pointer.read_bytes()
    pointer.unlink()
    pointer.write_bytes(raw)
    with pytest.raises(HandoffError):
        h.capture(allowed_new=set())


def test_new_ignore_cannot_hide_itself_and_other_new_files(case):
    _, root, h = case
    (root / "old").write_text("after")
    (root / "nested").mkdir()
    (root / "nested" / ".gitignore").write_text("*\n")
    (root / "nested" / "hidden").write_text("hidden by new rule")
    with pytest.raises(HandoffError, match="Ignore-rule"):
        h.capture(allowed_new=set())


@pytest.mark.parametrize('name,content', [
    ('.gitattributes', b'*.bin filter=lfs diff=lfs merge=lfs -text\n'),
    ('pointer', b'version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 10\n'),
])
def test_lfs_is_explicitly_unsupported(case, name, content):
    _, root, h = case
    (root / name).write_bytes(content)
    with pytest.raises(HandoffError, match='LFS'):
        h.capture(allowed_new={name})


def test_checkout_alias_is_canonicalized_for_private_git(case, tmp_path):
    _, root, old = case
    alias = tmp_path / 'checkout-alias'
    alias.symlink_to(root, target_is_directory=True)
    h = Handoff.begin(alias, tmp_path / 'alias-trusted', old.identity, expected_base=old.base)
    assert h.checkout == root.resolve()
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
    assert h.verify(operator_commit(h))


def test_operator_context_starts_with_base_git_index(case):
    """Before an agent inspects status, the private index must reflect HEAD.

    An absent index looks like a staged deletion, despite unchanged source.
    """
    import os
    import subprocess
    _, root, handoff = case
    (root / 'old').write_text('changed by editor')
    handoff.capture(allowed_new=set())
    ctx = handoff.operator_context()
    assert (ctx.private / 'index').is_file()
    env = dict(os.environ, GIT_DIR=str(ctx.admin), GIT_WORK_TREE=str(root),
               GIT_INDEX_FILE=str(ctx.private / 'index'),
               GIT_OBJECT_DIRECTORY=str(ctx.objects),
               GIT_ALTERNATE_OBJECT_DIRECTORIES=str(ctx.shared),
               GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1')
    p = subprocess.run(['git', 'status', '--short', '--untracked-files=no'], cwd=root,
                       env=env, capture_output=True, text=True, check=True)
    assert p.stdout.startswith(' M old'), p.stdout
