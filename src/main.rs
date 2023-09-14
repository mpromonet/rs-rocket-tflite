use std::env::args;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder, Result};

#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, world!"
}

fn main() {
    assert_eq!(args().len(), 2, "minimal <tflite model>");

    let filename = args().nth(1).unwrap();

    let model = FlatBufferModel::build_from_file(filename)?;
    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(&model, &resolver)?;
    let mut interpreter = builder.build()?;

    interpreter.allocate_tensors()?;

    println!("=== Pre-invoke Interpreter State ===");
    interpreter.print_state();

    interpreter.invoke()?;

    println!("\n\n=== Post-invoke Interpreter State ===");
    interpreter.print_state();
       
    rocket::ignite().mount("/", routes![index]).launch();
}