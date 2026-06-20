def get_random_sky_coord():
    """Generates a random point on the sky with uniform spherical distribution."""
    ra = random.uniform(0.0, 360.0)
    
    # Dec requires a sine-distribution to avoid 'clustering' at the poles
    z = random.uniform(-1.0, 1.0)
    dec = np.degrees(np.arcsin(z))
    
    return round(ra, 4), round(dec, 4)