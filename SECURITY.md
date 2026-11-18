# Security

Please report potential vulnerabilities through GitHub private vulnerability reporting when available. Include the affected version, impact, and reproducible steps using synthetic data. Keep credentials and production ticket content out of reports.

The default Compose setup exposes CHECK on loopback. For deployment beyond the local machine, disable demo mode, set an access token, and put CHECK behind an authenticated HTTPS reverse proxy. Treat the database and its backups as sensitive because they can contain conversations, knowledge documents, review history, and cloud credentials.
