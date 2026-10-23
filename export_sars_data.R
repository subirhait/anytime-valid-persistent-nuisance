# Export the SARS2003 data distributed with EpiEstim for the Python application script.
# Run once from the simulations directory (or change out_dir below).
# install.packages("EpiEstim")  # if needed

library(EpiEstim)
data("SARS2003", package = "EpiEstim")

out_dir <- "epi_data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

inc <- data.frame(incidence = as.numeric(SARS2003$incidence))
si  <- data.frame(si_distr = as.numeric(SARS2003$si_distr))

write.csv(inc, file.path(out_dir, "SARS2003_incidence.csv"), row.names = FALSE)
write.csv(si,  file.path(out_dir, "SARS2003_si_distr.csv"), row.names = FALSE)

message("Wrote SARS2003_incidence.csv and SARS2003_si_distr.csv to ", normalizePath(out_dir))
