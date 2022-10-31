import os 
import sys 

if __name__=='__main__':
    
    """Prepebd a stable diffusion command to a file with prompts
    
    WE HAVE:
    
    $ cat prompts.txt 
        alice in wonderland -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        madhatter -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        white rabbit tea party -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g        
        
    WE WANT: 
    $ cat prompts2.txt 
        halloween, spooky, dark, alice in wonderland -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        halloween, spooky, dark, madhatter -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g 
        halloween, spooky, dark, white rabbit tea party -H800 -W600 -s 50 -C 7.0 -A k_lms -n12 -g        
        
    DO IT LIKE THIS: 
    
    $ line_to_add="halloween, spooky, dark"
    $ python prepend_to_prompts.py prompts.txt prompts2.txt $line_to_add 
    
    
    THEN RUN STABLE DIFFUSION: 
        
    >> python scripts/invoke.py --from_file prompts2.txt 
    
    
    
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
            newline = line_to_add + " " + line
            f.write(newline)
            
        