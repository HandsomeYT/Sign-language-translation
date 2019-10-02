function read_dir(){
    IFS=$'\n'
    for file in `ls $1`       
    do
        if [ -d $1"/"$file ] 
        then
            read_dir $1"/"$file
        else
            a=${file%%.*}
            mkdir $a
            python read.py $file $a
        fi
    done
}   

read_dir $1
