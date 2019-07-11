# README

## To create a master table

Add all .csv files to a folder inside /company-data/features, for example /company-data/features/example_folder_for_data.

Then run
`python create_master_table.py {path_to_folder_with_csv_files}`

## There might files with wrong delimiter (if downloaded from periscope)

To get rid of those, you can run 
`python replace_delimiter.py {path_to_file} "{delimiter}"`

## Count rows of files in a folder

`python row_counter.py {path_to_file}`


