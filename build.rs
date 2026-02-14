use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let output = Command::new("python3-config")
        .args(["--embed", "--ldflags"])
        .output();

    let Ok(output) = output else {
        println!("cargo:warning=python3-config not found; libpython link flags were not injected");
        return;
    };

    if !output.status.success() {
        println!(
            "cargo:warning=python3-config --embed --ldflags failed; libpython link flags were not injected"
        );
        return;
    }

    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            if !path.is_empty() {
                println!("cargo:rustc-link-search=native={path}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
            }
        } else if let Some(lib) = token.strip_prefix("-l") {
            if !lib.is_empty() {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
    }
}
