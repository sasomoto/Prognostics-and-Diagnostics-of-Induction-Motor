    def preprocess_lvm(self, file_content):
        """Process LVM file content."""
        lines = file_content.splitlines()
        header = lines[23].strip().split('\t')
        header = [col for col in header if col != 'Comment']
        data = [line.strip().split('\t') for line in lines[24:]]
        data = [[value for i, value in enumerate(row) if header[i] != 'Comment'] for row in data]
        return pd.DataFrame(data, columns=header)

    def process_files(self, zip_file_path, output_dir):
        """Process all LVM files from zip."""
        with ZipFile(zip_file_path, 'r') as zip_ref:
            extract_path = os.path.join(output_dir, 'extracted')
            zip_ref.extractall(extract_path)
            root_path = os.path.join(extract_path, 'Run 3', 'No load')
            folders = ['Healthy', 'R0S25', 'R0S50', 'R0S75']
            
            for folder in folders:
                folder_path = os.path.join(root_path, folder)
                output_folder = os.path.join(output_dir, folder)
                os.makedirs(output_folder, exist_ok=True)
                
                files = sorted([f for f in os.listdir(folder_path) if f.endswith('.lvm')])
                for file_name in files:
                    with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                        df = self.preprocess_lvm(file.read())
                        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.csv")
                        df.to_csv(output_file, index=False)
                        print(f"Created CSV: {output_file}")

    def combine_csvs(self, input_dir, output_dir):
        """Combine CSV files for each category."""
        categories = ['Healthy', 'R0S25', 'R0S50', 'R0S75']
        os.makedirs(output_dir, exist_ok=True)
        
        for category in categories:
            folder_path = os.path.join(input_dir, category)
            output_file = os.path.join(output_dir, f"Total_{category}.csv")
            
            csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
            combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"Created combined CSV: {output_file}")
