---
title: Prisma
date: 2021-10-03
description: Prisma Summary
category: summary
type: notes
---

# Prisma

Prisma is an type-safe ORM that can query a database in node.js apps with either REST or GraphQL APIs.

`npx prisma init` to create `.env` which stores db connection info and `prisma/schema.prisma` which has the schema

add models into `schema.prisma`

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
}
```

apply migrations by running `npx prisma migrate dev --name <migrationName>`.

run `npx prisma generate` to generate the client code off the models. Re-run every time you update the model.
