import os 
import sys 

if __name__=='__main__':
    
    """Append a stable diffusion command to a file with prompts
    
    WE HAVE:
    
    $ cat prompts.txt 
        alice in wonderland 
        madhatter 
        white rabbit tea party 
        
    WE WANT: 

    $ cat prompts_with_commands.txt 
        alice in wonderland -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        madhatter -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        white rabbit tea party -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
       
        
    DO IT LIKE THIS: 
    
    $ line_to_add="-H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g "
    $ python add_to_prompts.py prompts.txt prompts_with_commands.txt $line_to_add 
    
    
    THEN RUN STABLE DIFFUSION: 
        
    >> python scripts/invoke.py --from_file prompts_with_commands.txt 
    
    
    
    """
    
    filename=sys.argv[1]
    newfilename=sys.argv[2]
    line_to_add=sys.argv[3:]
    line_to_add = " ".join(line_to_add)
    
    
    assert os.path.exists(filename)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    with open(newfilename, 'w') as f:
        for line in lines: 
            newline = line[:-1] + " " + line_to_add + "\n"
            f.write(newline)
            
        