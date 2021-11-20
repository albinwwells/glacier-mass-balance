import numpy as np

def emergence_pixels(vel_x_raw, vel_y_raw, vel_total, vel_min, vel_max, vel_depth_avg_factor, icethickness, pixel_size):
    """ Compute the emergence velocity using an ice flux approach
    """
    # ----- Quality Control Options ----------
    option_minvelocity = 0
    option_maxvelocity = 1
    #  > 1 (default) - apply limits specified in model input
    #  > 0 - do not apply limits
    option_border = 1
    #  > 1 (default) - apply the border to prevent errors
    #  > 0 - do not apply border, assumes errors near edges are avoided elsewhere



    # Modify vel_y by multiplying velocity by -1 such that matrix operations agree with flow direction
    #    Specifically, a negative y velocity means the pixel is flowing south.
    #    However, if you were to subtract that value from the rows, it would head north in the matrix.
    #    This is due to the fact that the number of rows start at 0 at the top.
    #    Therefore, multipylying by -1 aligns the matrix operations with the flow direction
    vel_y = vel_y_raw * -1 * vel_depth_avg_factor
    vel_x = vel_x_raw * vel_depth_avg_factor
    # Compute the initial volume
    volume_initial = icethickness * pixel_size**2
    # Quality control options:
    # Apply a border based on the max specified velocity to prevent errors associated with pixels going out of bounds
    if option_border == 1:
        border = int(vel_max / pixel_size)
        for r in range(vel_x.shape[0]):
            for c in range(vel_x.shape[1]):
                if (r < border) | (r >= vel_x.shape[0] - border) | (c < border) | (c >= vel_x.shape[1] - border):
                    vel_x[r,c] = 0
                    vel_y[r,c] = 0
    # Minimum/maximum velocity bounds
    if option_minvelocity == 1:
        vel_x[vel_total < vel_min] = 0
        vel_y[vel_total < vel_min] = 0
    if option_maxvelocity == 1:
        vel_x[vel_total > vel_max] = 0
        vel_y[vel_total > vel_max] = 0
    # Compute displacement in units of pixels
    vel_x_pix = vel_x / pixel_size
    vel_y_pix = vel_y / pixel_size
    # Compute the displacement and fraction of pixels moved for all columns (x-axis)
    # col_x1 is the number of columns to the closest pixel receiving ice [ex. 2.6 returns 2, -2.6 returns -2]
    #    int() automatically rounds towards zero
    col_x1 = vel_x_pix.astype(int)
    # col_x2 is the number of columns to the further pixel receiving ice [ex. 2.6 returns 3, -2.6 returns -3]
    #    np.sign() returns a value of 1 or -1, so it's adding 1 pixel away from zero
    col_x2 = (vel_x_pix + np.sign(vel_x_pix)).astype(int)
    # rem_x2 is the fraction of the pixel that remains in the further pixel (col_x2) [ex. 2.6 returns 0.6, -2.6 returns 0.6]
    #    np.sign() returns a value of 1 or -1, so multiplying by that ensures you have a positive value
    #    then when you take the remainder using "% 1", you obtain the desired fraction
    rem_x2 = np.multiply(np.sign(vel_x_pix), vel_x_pix) % 1
    # rem_x1 is the fraction of the pixel that remains in the closer pixel (col_x1) [ex. 2.6 returns 0.4, -2.6 returns 0.4]
    rem_x1 = 1 - rem_x2
    # Repeat the displacement and fraction computations for all rows (y-axis)
    row_y1 = vel_y_pix.astype(int)
    row_y2 = (vel_y_pix + np.sign(vel_y_pix)).astype(int)
    rem_y2 = np.multiply(np.sign(vel_y_pix), vel_y_pix) % 1
    rem_y1 = 1 - rem_y2
    # Compute the mass flux for each pixel
    volume_final = np.zeros(volume_initial.shape)
    for r in range(vel_x.shape[0]):
        for c in range(vel_x.shape[1]):
            volume_final[r+row_y1[r,c], c+col_x1[r,c]] = (
                volume_final[r+row_y1[r,c], c+col_x1[r,c]] + rem_y1[r,c]*rem_x1[r,c]*volume_initial[r,c])
            volume_final[r+row_y2[r,c], c+col_x1[r,c]] = (
                volume_final[r+row_y2[r,c], c+col_x1[r,c]] + rem_y2[r,c]*rem_x1[r,c]*volume_initial[r,c])
            volume_final[r+row_y1[r,c], c+col_x2[r,c]] = (
                volume_final[r+row_y1[r,c], c+col_x2[r,c]] + rem_y1[r,c]*rem_x2[r,c]*volume_initial[r,c])
            volume_final[r+row_y2[r,c], c+col_x2[r,c]] = (
                volume_final[r+row_y2[r,c], c+col_x2[r,c]] + rem_y2[r,c]*rem_x2[r,c]*volume_initial[r,c])
     # Check that mass is conserved (threshold = 0.1 m x pixel_size**2)
    # print('Mass is conserved?', abs(volume_final.sum() - volume_initial.sum()) < 0.1 * pixel_size**2)
     # MASS CONSERVATION THRESHOLD OF: 1E-5%
    assert (100*abs(volume_final.sum() - volume_initial.sum())/volume_initial.sum()) < 1e-5, 'Mass is not conserved'

    # Final ice thickness
    icethickness_final = volume_final / pixel_size**2
    # Emergence velocity
    emergence_velocity = icethickness_final - icethickness
    return emergence_velocity