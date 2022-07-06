%% ======================================================================== FULL DATA SET (288 obs per day) ===================

ds_full = NedocData(TI_full);
ds_full = ds_full.setToday(0.99);
ds_full = ds_full.setPPD(48)