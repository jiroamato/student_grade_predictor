.PHONY: all clean

all: reports/student_grade_predictor_report.html reports/student_grade_predictor_report.pdf

# download and extract data
data/raw/student-por.csv : src/download_data.py
	python src/download_data.py \
		--url="https://archive.ics.uci.edu/static/public/320/student+performance.zip" \
		--write-to=data/raw

# split data into train and test sets, preprocess data
# and save preprocessor
data/processed/student_train.csv data/processed/student_test.csv results/models/student_preprocessor.pickle : src/preprocess_data.py \
data/raw/student-por.csv
	python src/preprocess_data.py \
		--raw-data=data/raw/student-por.csv \
		--data-to=data/processed \
		--preprocessor-to=results/models \
		--seed=123

# perform eda and save plots
results/figures/target_distribution.png results/figures/correlation_heatmap.png : src/eda.py \
data/processed/student_train.csv
	python src/eda.py \
		--processed-training-data=data/processed/student_train.csv \
		--plot-to=results/figures

# train model, visualize tuning, and save plot and model
results/models/student_pipeline.pickle results/figures/student_tune_alpha.png results/models/best_params.csv : src/fit_student_predictor.py \
data/processed/student_train.csv \
results/models/student_preprocessor.pickle
	python src/fit_student_predictor.py \
		--training-data=data/processed/student_train.csv \
		--preprocessor=results/models/student_preprocessor.pickle \
		--pipeline-to=results/models \
		--plot-to=results/figures \
		--seed=123

# evaluate model on test data and save results
results/tables/test_scores.csv results/tables/top_coefficients.csv results/figures/prediction_error.png : src/evaluate_student_predictor.py \
data/processed/student_test.csv \
results/models/student_pipeline.pickle
	python src/evaluate_student_predictor.py \
		--test-data=data/processed/student_test.csv \
		--pipeline-from=results/models/student_pipeline.pickle \
		--tables-to=results/tables \
		--plot-to=results/figures \
		--seed=123

# build HTML and PDF report
reports/student_grade_predictor_report.html reports/student_grade_predictor_report.pdf : reports/student_grade_predictor_report.qmd \
results/figures/target_distribution.png \
results/figures/correlation_heatmap.png \
results/figures/student_tune_alpha.png \
results/figures/prediction_error.png \
results/tables/test_scores.csv \
results/tables/top_coefficients.csv \
results/models/best_params.csv
	quarto render reports/student_grade_predictor_report.qmd

# clean up analysis
clean :
	rm -f data/raw/.student.zip_old \
		data/raw/student.zip \
		data/raw/student.txt \
		data/raw/student-merge.R \
		data/raw/student-mat.csv \
		data/raw/student-por.csv
	rm -f data/processed/student_train.csv \
		data/processed/student_test.csv \
		data/processed/transformed_student_test.csv \
		data/processed/transformed_student_train.csv
	rm -f results/models/student_preprocessor.pickle \
		results/models/student_pipeline.pickle \
		results/models/best_params.csv
	rm -f results/figures/target_distribution.png \
		results/figures/correlation_heatmap.png \
		results/figures/student_tune_alpha.png \
		results/figures/prediction_error.png
	rm -f results/tables/test_scores.csv \
		results/tables/top_coefficients.csv
	rm -f reports/student_grade_predictor_report.html \
		reports/student_grade_predictor_report.pdf \
