use slepc_rs::{vec::PetscVec, world::SlepcWorld, Result};

fn main() -> Result<()> {
    // Parameters
    let n = 10;

    println!("Hello World");
    let world = SlepcWorld::initialize()?;

    let mut x = PetscVec::create(&world)?;
    x.set_sizes(None, Some(n))?;
    x.set_up()?;
    x.assembly_begin()?;
    x.assembly_end()?;
    x.set_random(None)?;

    let x_view = x.get_array_read()?;
    println!("{:?}", x_view.len());
    //println!("{:?}", x_view);
    for (i, x) in x_view.iter().enumerate() {
        println!("i {:?} {:?}", i, x);
    }

    Ok(())
}
