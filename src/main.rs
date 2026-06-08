fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

fn main() {
    println!("{}", greet("world"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greet_world() {
        assert_eq!(greet("world"), "Hello, world!");
    }

    #[test]
    fn greet_custom_name() {
        assert_eq!(greet("N-Trancerator"), "Hello, N-Trancerator!");
    }
}
