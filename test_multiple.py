import subprocess
    
datasets = ["edit-haggle", "edit-SFHH", "edit-observational", "edit-sp-hypertext", "edit-sp-workplace", \
             "edit-hs2011", "edit-hs2012","edit-hs2013" \
            , "edit-ia-reality-call", "edit-infectious", "edit-sewiki",\
            "edit-company-emails", "edit-ants1-1", "edit-ants1-2",\
             "edit-ants2-1", "edit-ants2-2",
              "edit-eu-email-2", "edit-eu-email-3", "edit-eu-email-4", "edit-sp-hospital"]


for d in datasets:
    command = ["python", "-u", "main_28oct.py", "-d", d, "--test"]
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: Evaluation failed for dataset {d}")
