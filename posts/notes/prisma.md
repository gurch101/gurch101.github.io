---
title: Prisma
date: 2021-10-03
description: Prisma Summary
category: summary
type: notes
---

# Prisma

Prisma is an type-safe ORM that can query a database in node.js apps with either REST or GraphQL APIs.

1. `npm install --save-dev prisma` and `npm install @prisma/client`

2. `npx prisma init` to create `.env` which stores db connection info and `prisma/schema.prisma` which has the schema.

3. Set the provider in `schema.prisma` to sqlite/postgresql, etc.

4. Set the `DATABASE_URL` in `.env`. (for sqlite, `file:./dev.db)

5. add models into `schema.prisma`

```sh
model User {
  id    Int     @default(autoincrement()) @id
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id        Int      @default(autoincrement()) @id
  title     String
  content   String?
  published Boolean? @default(false)
  author    User?    @relation(fields: [authorId], references: [id])
  authorId  Int?
  createDate DateTime @default(now())
}
```

6. apply migrations by running `npx prisma migrate dev --name <migrationName>`.

7. run `npx prisma generate` to generate the client code off the models in the schema. Re-run every time you update the model.

### Client

```ts
import { PrismaClient } from "@prisma/client";

// only create one instance in your application. Each client manages a connection pool of size (cpus * 2) + 1.
// automatically connects lazily on first request
const prisma = new PrismaClient();
```

https://www.prisma.io/docs/concepts/components/prisma-client/working-with-prismaclient/connection-pool

### Migrations

Update schema then run `npx prisma migrate dev` to create (if needed) and apply migrations. Use this command in development environments.

To apply migrations in staging or production, run `npx prisma migrate deploy`.

To make sure the database is in sync with the prisma schema (for a pre-existing db, for ex), run `prisma db pull`

To create a migration (based on schema or empty file) without applying it, run `npx prisma migrate dev --create-only`. This is useful if you need to modify the migration file (to handle use cases not handled by prisma schema files - adding extensions, stored procedures, triggers, views, partial indexes) before applying it. This is also useful if you need to rename fields since a rename is a CREATE + DROP - you should manually modify the migration to `ALTER TABLE RENAME`.

To baseline a migration (mark it as applied) for pre-existing databases, run `prisma migrate resolve --applied <migration name>`.

Migrations are applied in the order they are created.

`prisma migrate diff` can diff two schema sources (file/db) and outputs the difference to a sql script.

### Seed Database

1. Create `prisma/seed.ts` file

```ts
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  await prisma.user.create({
    data: {},
  });
}

main()
  .catch((e) => console.log(e))
  .finally(async () => await prisma.$disconnect());
```

2. Update `package.json`

```json
{
  "prisma": {
    "seed": "ts-node prisma/seed.ts"
  }
}
```

3. Apply with `npx prisma db seed`, `npx prisma migrate dev`, or `npx prisma migrate reset`

### Limitations

- cascades aren't supported by prisma schema, need to manually maintain in migration files.

### Nestjs integration

```ts
import { INestApplication, Injectable, OnModuleInit } from "@nestjs/common";
import { PrismaClient } from "@prisma/client";

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit {
  async onModuleInit() {
    await this.$connect();
  }

  async enableShutdownHooks(app: INestApplication) {
    this.$on("beforeExit", async () => {
      await app.close();
    });
  }
}
```
